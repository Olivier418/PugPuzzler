"""
Core data structures for incremental connectivity tracking, standalone
from Solver/Game so they can be tested against the real Board/Lattice
classes without needing constants.py, partitioning.py, blocks.py, or
classes.py.
"""

from __future__ import annotations

import numpy as np


class CompactBoard:
    """Maps a board's *real* cells (board.cells == True) to a dense
    0..n-1 range, dropping OUTSIDE_BOARD cells entirely, and precomputes
    the lattice-adjacency graph over that dense range.
    """

    def __init__(self, board):
        cells_flat = board.cells.ravel()
        real_flat_indices = np.flatnonzero(cells_flat)

        self.n = real_flat_indices.size
        self.compact_to_flat = real_flat_indices                      # (n,)
        self.flat_to_compact = np.full(cells_flat.size, -1, dtype=np.int64)
        self.flat_to_compact[real_flat_indices] = np.arange(self.n)

        self.adjacency = self._build_adjacency(board)

    def _build_adjacency(self, board) -> list[np.ndarray]:
        """adjacency[i] = compact indices of every real cell adjacent to
        compact cell i. Built from lattice.unit_vectors directly (the
        same offsets neighbor_structure encodes as a stencil), since we
        need per-axis bounds/validity checks either way.
        """
        shape = board.shape
        coords = np.array(np.unravel_index(self.compact_to_flat, shape)).T  # (n, ndim)

        adjacency: list[list[int]] = [[] for _ in range(self.n)]

        for v in board.lattice.unit_vectors:
            neighbor_coords = coords + v
            in_bounds = np.all(
                (neighbor_coords >= 0) & (neighbor_coords < np.array(shape)),
                axis=1,
            )

            # Only touch board.cells for the in-bounds subset -- otherwise
            # negative indices would wrap around and silently look valid.
            idxs = np.flatnonzero(in_bounds)
            nb_tuple = tuple(neighbor_coords[idxs].T)
            valid_local = np.zeros(self.n, dtype=bool)
            valid_local[idxs] = board.cells[nb_tuple]

            src = np.flatnonzero(valid_local)
            dst_flat = np.ravel_multi_index(tuple(neighbor_coords[src].T), shape)
            dst_compact = self.flat_to_compact[dst_flat]

            for s, d in zip(src, dst_compact):
                adjacency[s].append(int(d))

        # Plain Python lists, not numpy arrays: everywhere this gets used
        # is fine-grained per-element iteration in Python loops (the
        # search's hot path), and numpy arrays are markedly slower there
        # -- confirmed ~8x slower to iterate one Python-level element at a
        # time versus a plain list, since each access boxes a numpy
        # scalar. Numpy earns its keep in the vectorized construction
        # above; it doesn't in per-cell traversal below.
        return adjacency


class ComponentTracker:
    """Incrementally tracks connected components of the *available*
    (unoccupied) compact cells.

    Cells only ever leave the available set (a block gets placed,
    possibly splitting a component) or rejoin it in exactly the reverse
    of a prior split (backtracking) -- so every removal is paired with
    an exact undo, never a fresh merge from scratch. That asymmetry is
    what makes the rollback cheap: we never need the general "reconnect
    two arbitrary components" case.

    Membership is stored as a plain list per component ("arr") plus a
    global cell -> position-within-its-arr map ("cell_pos"), NOT a
    Python set. A Python set forces any removal of a few elements to pay
    for touching the whole set (building a new one, or copying before an
    in-place difference_update -- both cost O(len(big set)), confirmed
    empirically; there's no cheap direction). An array with swap-and-pop
    removes a cell in true O(1): swap it with the last element, shrink
    by one. That's what actually makes "placing a block only costs
    O(block size), not O(component size)" true in practice, not just in
    complexity-notation theory.
    """

    def __init__(self, compact_board: CompactBoard, available_mask: np.ndarray):
        self.adjacency = compact_board.adjacency
        self.n = compact_board.n

        # Plain Python list, not a numpy array -- same reasoning as
        # adjacency: this is read/written one cell at a time throughout
        # the hot path, never in bulk, so a numpy array here just adds
        # scalar-boxing overhead on every access for no vectorization
        # benefit.
        self.component_id: list[int] = [-1] * self.n
        self.members: dict[int, list[int]] = {}
        self.cell_pos: dict[int, int] = {}
        self._next_id = 0
        self._undo_stack: list[dict] = []

        self._seed_components(available_mask)

    def _new_id(self) -> int:
        cid = self._next_id
        self._next_id += 1
        return cid

    def _seed_components(self, available_mask: np.ndarray):
        visited = np.zeros(self.n, dtype=bool)
        for start in np.flatnonzero(available_mask):
            start = int(start)
            if visited[start]:
                continue
            comp_id = self._new_id()
            stack = [start]
            visited[start] = True
            members: list[int] = []
            while stack:
                cell = stack.pop()
                self.cell_pos[cell] = len(members)
                members.append(cell)
                self.component_id[cell] = comp_id
                for nb in self.adjacency[cell]:
                    if available_mask[nb] and not visited[nb]:
                        visited[nb] = True
                        stack.append(nb)
            self.members[comp_id] = members

    def _swap_remove(self, cell: int):
        """O(1): remove one cell from its component's member array by
        swapping the last element into its slot and shrinking."""
        cid = self.component_id[cell]
        arr = self.members[cid]
        pos = self.cell_pos.pop(cell)
        last_pos = len(arr) - 1
        if pos != last_pos:
            last_cell = arr[last_pos]
            arr[pos] = last_cell
            self.cell_pos[last_cell] = pos
        arr.pop()
        self.component_id[cell] = -1

    def remove_cells(self, cells: np.ndarray) -> list[int]:
        """Mark `cells` (compact indices) as occupied -- a block was just
        placed there. Splits the owning component if needed and returns
        the list of freshly-created component ids (empty if no split
        occurred). Every cell in `cells` is assumed to belong to the same
        pre-existing component: true whenever the caller only ever places
        blocks (themselves lattice-connected shapes) entirely within
        cells already known to be available.

        Membership tests during the search below go through
        `component_id` (an O(1) array lookup) rather than any per-call
        set/copy of the component's members -- that's what keeps this
        from paying O(component size) except in the branch that
        genuinely needs it (an actual split, where enumerating every
        resulting fragment is real, unavoidable output work).

        Fast path: before removal, the whole component was one connected
        piece (tracker invariant). Any two remaining cells whose only old
        path ran through a removed cell must reduce to a path between
        "seed" cells -- remaining cells directly adjacent to a removed
        cell. So if a BFS from one seed reaches every other seed without
        exhausting the remaining region, the whole remaining region is
        *provably* still one component, however big it is.
        """
        owning_id = self.component_id[cells[0]]
        removed = set(int(c) for c in cells)

        frame = {
            "owning_id": owning_id,
            "removed": removed,
            "next_id_before": self._next_id,
            "new_ids": [],
            "swaps": [],   # (cell, cid, pos, swapped_in_cell_or_None), undo for the no-split path
            "split": None,  # filled in only if a split happens
            "emptied": False,  # True if the no-split path consumed the whole component
        }

        arr = self.members[owning_id]
        if len(removed) == len(arr):
            # every cell in the component just got placed
            return self._finish_no_split(removed, owning_id, frame)

        seeds = set()
        for c in removed:
            for nb in self.adjacency[c]:
                if self.component_id[nb] == owning_id and nb not in removed:
                    seeds.add(nb)

        if len(seeds) <= 1:
            # Zero or one entry point into the rest of the component --
            # can't possibly have split. Just excise the removed cells.
            return self._finish_no_split(removed, owning_id, frame)

        seeds_list = list(seeds)
        root = seeds_list[0]
        pending_seeds = set(seeds_list[1:])

        visited = {root}
        first_frag = [root]
        stack = [root]
        while stack:
            cell = stack.pop()
            for nb in self.adjacency[cell]:
                if (
                    self.component_id[nb] == owning_id
                    and nb not in removed
                    and nb not in visited
                ):
                    visited.add(nb)
                    first_frag.append(nb)
                    stack.append(nb)
                    pending_seeds.discard(nb)
            if not pending_seeds:
                break

        if not pending_seeds:
            # Every seed reached -- proven connected, however big the
            # rest of the component is. Stop here.
            return self._finish_no_split(removed, owning_id, frame)

        # Genuine split: now we do need to enumerate every fragment --
        # real, unavoidable output work at this point. `first_frag` is
        # already complete (the while loop only exits here once `stack`
        # is exhausted), so flood only whatever's left.
        rest = [c for c in arr if c not in removed and c not in visited]
        fragments = [first_frag] + self._flood_fragments(rest)
        fragments.sort(key=len, reverse=True)

        frame["split"] = {
            "old_arr": list(arr),
            "old_cell_pos": {c: self.cell_pos[c] for c in arr if c in self.cell_pos},
        }

        self.members[owning_id] = fragments[0]
        for i, c in enumerate(fragments[0]):
            self.component_id[c] = owning_id
            self.cell_pos[c] = i

        for frag in fragments[1:]:
            new_id = self._new_id()
            self.members[new_id] = frag
            for i, c in enumerate(frag):
                self.component_id[c] = new_id
                self.cell_pos[c] = i
            frame["new_ids"].append(new_id)

        for c in removed:
            self.component_id[c] = -1
            self.cell_pos.pop(c, None)

        self._undo_stack.append(frame)
        return list(frame["new_ids"])

    def _finish_no_split(self, removed, owning_id, frame):
        for c in removed:
            self._record_and_remove(c, frame)
        if not self.members[owning_id]:
            # The component was fully consumed -- it isn't "live"
            # anymore, so it must not linger in `members` as a phantom
            # zero-size entry (that would make num_components overcount
            # what the search actually has left to work with).
            del self.members[owning_id]
            frame["emptied"] = True
        self._undo_stack.append(frame)
        return []

    def _record_and_remove(self, cell: int, frame: dict):
        cid = self.component_id[cell]
        arr = self.members[cid]
        pos = self.cell_pos[cell]
        last_pos = len(arr) - 1
        swapped_in = arr[last_pos] if pos != last_pos else None
        frame["swaps"].append((cell, cid, pos, swapped_in))
        self._swap_remove(cell)

    def restore(self):
        """Undo the most recent remove_cells call exactly."""
        frame = self._undo_stack.pop()
        self._next_id = frame["next_id_before"]

        if frame["split"] is not None:
            owning_id = frame["owning_id"]
            for cid in frame["new_ids"]:
                del self.members[cid]
            self.members[owning_id] = frame["split"]["old_arr"]
            for c in frame["split"]["old_arr"]:
                self.component_id[c] = owning_id
            self.cell_pos.update(frame["split"]["old_cell_pos"])
            return

        # No split occurred: undo each swap-removal in reverse order.
        # If the component was deleted for being fully consumed, its key
        # has to exist again before any cell can be appended back into it.
        owning_id = frame["owning_id"]
        if frame["emptied"]:
            self.members[owning_id] = []

        # Reversing a swap means putting `cell` back at `pos`; if a
        # `swapped_in` cell had been moved into that slot, it needs to
        # move back to the end (where it was before the swap).
        for cell, cid, pos, swapped_in in reversed(frame["swaps"]):
            arr = self.members[cid]
            if swapped_in is None:
                arr.append(cell)
                self.cell_pos[cell] = pos
            else:
                arr.append(swapped_in)          # swapped_in goes back to the end
                self.cell_pos[swapped_in] = len(arr) - 1
                arr[pos] = cell
                self.cell_pos[cell] = pos
            self.component_id[cell] = cid

    def _flood_fragments(self, cells: list[int]) -> list[list[int]]:
        """Flood-fill `cells` into connected fragments."""
        remaining = set(cells)
        visited: set[int] = set()
        fragments = []
        for start in cells:
            if start in visited:
                continue
            stack = [start]
            visited.add(start)
            frag = []
            while stack:
                cell = stack.pop()
                frag.append(cell)
                for nb in self.adjacency[cell]:
                    if nb in remaining and nb not in visited:
                        visited.add(nb)
                        stack.append(nb)
            fragments.append(frag)
        return fragments