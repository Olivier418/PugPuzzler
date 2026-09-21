from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

from constants import EMPTY, UNPLACED

if TYPE_CHECKING:
    # Only for the type hint below -- never imported at runtime, so this
    # doesn't create a circular import with state.py (which imports
    # Solver at the top of the file).
    from .state import State


# block_count entry for a block that has already been placed. Large enough that
# `block_count.min() == _PLACED` means "nothing left to place".
_PLACED = np.iinfo(np.int64).max // 4


class Solver:
    """Exhaustive exact-cover solver for a State (or Puzzle): every block
    must be placed and every open cell covered exactly once.

    The search branches on the open cell covered by the fewest live
    placements (across all unplaced blocks) -- the most constrained cell --
    and tries every placement that covers it. Every solution covers that
    cell exactly once, so those branches partition the solutions: nothing
    is missed and nothing is found twice.

    Only reads State's public surface (board, blocks, placements,
    chosen_placement_idx, grid, place(), remove(), copy()) -- no
    knowledge of Puzzle's letter-grid loading or any other
    subclass-specific behavior is required.
    """

    def __init__(self, state: "State"):
        # A private working copy: place()/remove() mutate grid and
        # chosen_placement_idx in lockstep as the search backtracks, but
        # there's no reason that churn should be visible on (or tied to)
        # the caller's own state. self.state.copy() is cheap -- setup
        # (board/blocks/placements) is shared by reference, only the
        # small per-instance grid + chosen_placement_idx dict are copied.
        self.state = state.copy()

        self.placement_cells = self.state.placement_cells

        # Blocks the state already considers placed (e.g. letters baked
        # into a Puzzle's starting grid) are fixed and excluded from the
        # search entirely.
        self.preplaced = {
            idx for idx, placement_idx in self.state.chosen_placement_idx.items()
            if placement_idx != UNPLACED
        }

        # The search relies on the unplaced blocks filling the open cells
        # exactly (that is what makes a cell with no placement left a dead
        # end, and what lets it branch on any one cell). Setup._validate
        # guarantees it: block cells == board cells, and preplaced blocks
        # only ever fill whole placements.

        # A placement's row index in self.placement_cells is its identity, and
        # nothing here ever reorders those rows: a seed only changes the
        # priorities that break ties (see solve()). So every table indexed
        # by placement is built once and never goes stale.
        self._build_flat_tables()

    def _build_flat_tables(self):
        """Every block's placements laid end to end in one array, so the
        search state is a handful of flat numpy arrays instead of
        per-block structures -- one array op per node, not a Python loop
        over blocks.

        A placement's *global id* is its row in `_placement_cells_flat`; block position p
        (its index in `_block_ids`) owns ids `_block_start[p]:_block_start[p + 1]`, so a
        block-local placement index is `gid - _block_start[p]`. Blocks differ in
        size, so short rows are padded with the extra cell `n_cells`, which
        is never unavailable and never counted as covered by anything that
        matters."""
        n_cells = self.state.setup.n_cells
        self._block_ids = list(self.placement_cells)
        sizes = [self.placement_cells[b].shape[0] for b in self._block_ids]
        self._block_start = np.concatenate([[0], np.cumsum(sizes)]).astype(np.intp)
        n_total = int(self._block_start[-1])
        kmax = max(
            (arr.shape[1] for arr in self.placement_cells.values() if arr.shape[0]), default=1
        )

        P = np.full((n_total, kmax), n_cells, dtype=np.intp)
        for pos, block_idx in enumerate(self._block_ids):
            arr = self.placement_cells[block_idx]
            if arr.shape[0]:
                P[self._block_start[pos]:self._block_start[pos + 1], :arr.shape[1]] = arr
        self._placement_cells_flat = P
        self._block_of = np.repeat(np.arange(len(self._block_ids)), sizes)

        # cell -> global ids of every placement (of any block) covering it.
        flat = P.ravel()
        real = flat < n_cells
        gids = np.repeat(np.arange(n_total), kmax)[real]
        by_cell = np.argsort(flat[real], kind="stable")
        bounds = np.concatenate([[0], np.cumsum(np.bincount(flat[real], minlength=n_cells))])
        self._cell_lists = [gids[by_cell][bounds[c]:bounds[c + 1]] for c in range(n_cells)]

        # gid -> ids of every placement overlapping it (its own block's
        # included), filled in on first use by _overlapping.
        self._overlap_cache = [None] * n_total

    def _overlapping(self, gid) -> np.ndarray:
        """Global ids of every placement sharing a cell with `gid` (itself
        and its own block's included), each listed once. Pure geometry, so
        computed on first request and kept -- a search revisits the same
        placements over and over, and this turns "which placements does
        this piece invalidate" into a single lookup."""
        ids = self._overlap_cache[gid]
        if ids is None:
            n_cells = self.state.setup.n_cells
            row = self._placement_cells_flat[gid]
            ids = np.unique(np.concatenate([self._cell_lists[c] for c in row[row < n_cells]]))
            self._overlap_cache[gid] = ids
        return ids

    def _root_node(self):
        """The search's starting node: every placement of a block that
        still has to be placed and fits the cells that are still empty,
        as `(live, block_count, cell_count)`, or None if some such block has
        no placement at all.

        `live` is a bool array over global ids. `block_count[p]` is block
        position p's live placement count, or _PLACED once it is placed.
        `cell_count[c]` is how many live placements cover cell c (index
        n_cells is the padding cell). Every later node is derived from its
        parent by _place, never rebuilt."""
        n_cells = self.state.setup.n_cells
        unplaced = np.array([idx not in self.preplaced for idx in self._block_ids], dtype=bool)
        filled = np.append(self.state.grid != EMPTY, False)  # padding cell never blocks

        live = unplaced[self._block_of] & ~filled[self._placement_cells_flat].any(axis=1)
        block_count = np.where(
            unplaced, np.bincount(self._block_of[live], minlength=unplaced.size), _PLACED
        )
        if block_count.min() == 0:
            return None
        cell_count = np.bincount(self._placement_cells_flat[live].ravel(), minlength=n_cells + 1)
        return live, block_count, cell_count

    def _place(self, live, block_count, cell_count, pos, gid):
        """The child node after placing the placement with global id `gid`
        (of block position `pos`): that block's other placements leave the
        pool, and so does every placement (of any block) overlapping it.
        Everything is derived from the parent's arrays by touching only the
        placements that actually change (see _overlapping), rather than
        re-testing anything.

        The parent's arrays are left alone (siblings share them); returns
        (live, block_count, cell_count) for the child, or None if some other
        block is left with no placement."""
        lo, hi = self._block_start[pos], self._block_start[pos + 1]
        new_live = live.copy()

        own = np.flatnonzero(live[lo:hi]) + lo
        new_live[lo:hi] = False

        # Already-dead ids (including the whole placed block) drop out here.
        conflicts = self._overlapping(gid)
        hit = conflicts[new_live[conflicts]]
        new_live[hit] = False

        new_block_count = block_count - np.bincount(self._block_of[hit], minlength=block_count.size)
        new_block_count[pos] = _PLACED
        if new_block_count.min() == 0:
            return None

        gone = np.concatenate((self._placement_cells_flat[own], self._placement_cells_flat[hit])).ravel()
        new_cell_count = cell_count - np.bincount(gone, minlength=cell_count.size)
        return new_live, new_block_count, new_cell_count

    def _cell_branch_candidates(self, live, open_cells, cell_count):
        """The open cell covered by the fewest live placements
        (`cell_count`, across every unplaced block) and every placement
        covering it, as `[(block_position, global_id), ...]` in visit
        order. None if some open cell has no live placement left -- a dead
        end, since every open cell has to be covered.

        Visit order is lexicographic: each candidate's cells' live
        placement counts, sorted ascending, compared like tuples -- so a
        candidate covering the scarcest cells comes first, with no weights
        to tune. Whatever is still tied (equal count vectors, or several
        equally scarce cells) is decided by the per-solve priorities that
        `solve` draws from the seed (identity order when unseeded)."""
        n_cells = self.state.setup.n_cells
        counts = cell_count[:n_cells]
        if np.any(open_cells & (counts == 0)):
            return None

        # One argmin picks the scarcest open cell, priority breaking ties.
        # Live placements only ever cover open cells, so a closed cell's
        # count is 0 -- it gets the largest key instead of winning.
        pad = int(counts.max()) + 1
        cell = int(np.argmin(np.where(
            open_cells, counts * n_cells + self._cell_priority, pad * n_cells
        )))

        gids = self._cell_lists[cell]
        gids = gids[live[gids]]
        rows = self._placement_cells_flat[gids]
        keys = cell_count[rows]
        keys[rows == n_cells] = pad  # padding sorts after any real count
        keys.sort(axis=1)

        # lexsort treats its *last* key as primary.
        gids = gids[np.lexsort((self._priority[gids], *keys.T[::-1]))]
        return list(zip(self._block_of[gids].tolist(), gids.tolist()))

    def solve(self, seed: int = None, time_limit: float = math.inf, max_solutions: float = math.inf):
        """Yield every solution as a fully placed copy of the state, or --
        if a limit is hit first -- just the ones found by then.

        `time_limit` (seconds, counted from the first `next()`) and
        `max_solutions` both default to infinity; the search stops as soon
        as either is reached. The time is checked once per search node, so
        it can overshoot by about one node.

        `seed` randomises the order solutions are found in without
        changing the set: it draws one random priority per placement and
        per cell, used only to break ties the search's own ordering
        leaves open. Unseeded, ties fall to placement/cell index order.
        Nothing is ever reordered or shared, so a seed can't disturb
        anything indexed by placement."""
        n_total = self._placement_cells_flat.shape[0]
        n_cells = self.state.setup.n_cells
        if seed is None:
            self._priority = np.arange(n_total)
            self._cell_priority = np.arange(n_cells)
        else:
            rng = np.random.default_rng(seed)
            self._priority = rng.permutation(n_total)
            self._cell_priority = rng.permutation(n_cells)

        deadline = time.perf_counter() + time_limit
        found = 0
        stopped = max_solutions <= 0

        def search(live, block_count, cell_count, open_cells):
            """`live`/`block_count`/`cell_count` are a valid node (see
            _root_node), already pruned against `open_cells`, and
            owned by this call."""
            nonlocal found, stopped
            if stopped or time.perf_counter() >= deadline:
                # Sticky, so every caller up the stack unwinds too (each
                # undoing its own placement on the way out).
                stopped = True
                return

            if block_count.min() == _PLACED:
                # A full State snapshot, independent of self.state (which
                # keeps getting mutated as the search backtracks further).
                yield self.state.copy()
                found += 1
                stopped = found >= max_solutions
                return

            candidates = self._cell_branch_candidates(live, open_cells, cell_count)
            if candidates is None:
                return

            for pos, gid in candidates:
                block_idx = self._block_ids[pos]
                placement_idx = int(gid - self._block_start[pos])
                placement = self.placement_cells[block_idx][placement_idx]  # (count,) flat indices

                child = self._place(live, block_count, cell_count, pos, gid)
                if child is None:
                    continue

                self.state.place_unchecked(block_idx, placement_idx)
                open_cells.flat[placement] = False

                yield from search(*child, open_cells)

                self.state.remove_unchecked(block_idx)
                open_cells.flat[placement] = True

        node = self._root_node()
        if node is not None:
            yield from search(*node, self.state.grid == EMPTY)