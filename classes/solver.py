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

    By default the search branches on the open cell covered by the fewest
    live placements (across all unplaced blocks) -- the most constrained
    cell -- and tries every placement that covers it. Every solution
    covers that cell exactly once, so those branches partition the
    solutions: nothing is missed and nothing is found twice.

    Where the still-open region is symmetric it branches on a block
    instead, and searches only one placement per symmetry orbit; see
    solve()'s `branching`/`symmetry`/`up_to_symmetry`. That only happens
    near the root -- the group dies as soon as a placement breaks it --
    so the cell branch above is still what runs essentially everywhere.

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

        # Setup.placement_lookup, bound by solve() only when a mode can
        # actually use symmetry -- so a solve that can't never builds it.
        self._placement_lookup = None

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

    def _order_gids(self, gids, cell_count):
        """`gids` in visit order: lexicographic by each placement's cells'
        live placement counts, sorted ascending and compared like tuples
        -- so a placement covering the scarcest cells comes first, with no
        weights to tune. Whatever is still tied (equal count vectors) is
        decided by the per-solve priorities `solve` draws from the seed
        (identity order when unseeded).

        Shared by the cell branch, the block branch and the symmetry
        orbit sweep, so all three agree on which candidate is "best" --
        which is what makes the orbit representative the best-ranked
        member of its orbit rather than an arbitrary one."""
        n_cells = self.state.setup.n_cells
        rows = self._placement_cells_flat[gids]
        keys = cell_count[rows]
        keys[rows == n_cells] = int(cell_count[:n_cells].max()) + 1  # padding sorts last
        keys.sort(axis=1)
        # lexsort treats its *last* key as primary.
        return gids[np.lexsort((self._priority[gids], *keys.T[::-1]))]

    def _dead_end(self, open_cells, cell_count):
        """True if some open cell has no live placement left -- nothing
        can ever cover it, so this node is finished."""
        n_cells = self.state.setup.n_cells
        return bool(np.any(open_cells & (cell_count[:n_cells] == 0)))

    def _cell_branch_candidates(self, live, open_cells, cell_count):
        """The open cell covered by the fewest live placements
        (`cell_count`, across every unplaced block) and every placement
        covering it, as `[(block_position, global_id), ...]` in visit
        order. None if the node is a dead end.

        Every solution covers that cell exactly once, so these branches
        partition the solutions."""
        n_cells = self.state.setup.n_cells
        counts = cell_count[:n_cells]
        if self._dead_end(open_cells, cell_count):
            return None

        # One argmin picks the scarcest open cell, priority breaking ties.
        # Live placements only ever cover open cells, so a closed cell's
        # count is 0 -- it gets the largest key instead of winning.
        pad = int(counts.max()) + 1
        cell = int(np.argmin(np.where(
            open_cells, counts * n_cells + self._cell_priority, pad * n_cells
        )))

        gids = self._cell_lists[cell]
        gids = self._order_gids(gids[live[gids]], cell_count)
        return list(zip(self._block_of[gids].tolist(), gids.tolist()))

    def _block_branch_candidates(self, live, block_count, cell_count, open_cells):
        """The still-unplaced block with the fewest live placements, as
        `(block_position, global_ids in visit order)`. None if the node is
        a dead end.

        Every solution places that block exactly once, so these branches
        partition the solutions -- and unlike the cell branch, the set of
        them is carried onto itself by any symmetry of the open region,
        which is what lets `solve` collapse it into orbits (see
        _orbits)."""
        if self._dead_end(open_cells, cell_count):
            return None
        pos = int(np.argmin(block_count))
        lo, hi = self._block_start[pos], self._block_start[pos + 1]
        gids = np.flatnonzero(live[lo:hi]) + lo
        if gids.size == 0:
            return None
        return pos, self._order_gids(gids, cell_count)

    # ---- symmetry -------------------------------------------------------
    # A symmetry is an `image`: a permutation of compact cell indices that
    # is the identity outside the region it was computed for. `solve` asks
    # Setup for the root's group once; every group used deeper is a
    # subgroup of it, found by filtering (see _orbits' stabilizer) rather
    # than by searching the geometry again.

    def _image_placement_idx(self, block_idx, placement_idx, image):
        """The placement of the same block that `image` maps placement
        `placement_idx` onto. Guaranteed to hit -- see
        Setup.placement_lookup."""
        cells = self.placement_cells[block_idx][placement_idx]
        return self._placement_lookup[block_idx][np.sort(image[cells]).tobytes()]

    def _placement_image(self, pos, gid, image):
        """_image_placement_idx in global-id space."""
        lo = self._block_start[pos]
        return self._image_placement_idx(self._block_ids[pos], int(gid - lo), image) + lo

    def _orbits(self, pos, gids, images):
        """Group `gids` (block position `pos`'s live placements, all lying
        inside the symmetric region) into orbits under `images`, yielding
        `(representative, [image onto each other member], stabilizer)` per
        orbit.

        `gids` is in visit order, so the first unvisited one represents
        its orbit -- and since `images` forms a group, its orbit under
        those images is the whole orbit, so the image carrying it onto
        each other member falls out of the same sweep.

        `stabilizer` is the subgroup fixing the representative's cells,
        i.e. the symmetries still live once it is placed. It always
        contains the identity, so it is never empty; anything more means
        the child region is still symmetric and the reduction nests."""
        visited = set()
        for gid in gids.tolist():
            if gid in visited:
                continue
            orbit = {}
            stabilizer = []
            for image in images:
                other = self._placement_image(pos, gid, image)
                if other == gid:
                    stabilizer.append(image)
                orbit.setdefault(other, image)
            visited |= orbit.keys()
            yield gid, [img for other, img in orbit.items() if other != gid], stabilizer

    def _transform(self, solution, image):
        """A completed State/Puzzle snapshot, transformed by `image`: the
        mirror/rotation of `solution` in the branch the search skipped,
        which the symmetry guarantees is a valid completion once
        `solution` is.

        Transforming the *whole* grid is safe even though the symmetry is
        only about one node's open region: the image is the identity
        outside that region, so preplaced blocks and everything an
        ancestor decided come back out untouched -- including their
        placement indices, which the lookup below resolves to
        themselves."""
        transformed = solution.copy()
        transformed.grid = np.empty_like(solution.grid)
        transformed.grid[image] = solution.grid
        transformed.chosen_placement_idx = {
            idx: (self._image_placement_idx(idx, p, image) if p != UNPLACED else UNPLACED)
            for idx, p in solution.chosen_placement_idx.items()
        }
        return transformed

    def solve(
        self,
        seed: int = None,
        time_limit: float = math.inf,
        max_solutions: float = math.inf,
        branching: str = "balanced",
        symmetry: bool = True,
        up_to_symmetry: bool = False,
    ):
        """Yield every solution as a fully placed copy of the state, or --
        if a limit is hit first -- just the ones found by then.

        `time_limit` (seconds, counted from the first `next()`) and
        `max_solutions` both default to infinity; the search stops as soon
        as either is reached. The time is checked once per search node, so
        it can overshoot by about one node. Derived solutions (see
        `symmetry`) count towards `max_solutions` like any other.

        `seed` randomises the order solutions are found in without
        changing the set: it draws one random priority per placement and
        per cell, used only to break ties the search's own ordering
        leaves open. Unseeded, ties fall to placement/cell index order.
        Nothing is ever reordered or shared, so a seed can't disturb
        anything indexed by placement.

        `branching` picks what each node branches on:
          - "cell": the open cell covered by the fewest live placements,
            trying every placement covering it. The strongest heuristic,
            but its branch set is not carried onto itself by a symmetry
            of the open region, so it cannot use `symmetry` at all.
          - "block": the unplaced block with the fewest live placements,
            trying every one of them. Markedly weaker on its own -- it is
            here as the symmetry-compatible branch and as a benchmark
            baseline, not as a way to solve puzzles.
          - "balanced" (default): block while the open region still has a
            live symmetry group, cell once it doesn't. Since the group
            dies within a few levels of the root on almost every branch,
            this is cell-branching everywhere that matters, with the
            symmetry collapsed where it exists.

        `symmetry` turns the orbit reduction on: at a node whose open
        region has a nontrivial symmetry group, the branching block's
        placements are grouped into orbits, only one representative per
        orbit is searched, and the other members' solutions are produced
        by transforming the representative's. The solution *set* is
        unchanged; the order is not (a solution is followed by its
        images). It is a no-op with `branching="cell"`.

        `up_to_symmetry` yields only the representatives -- one solution
        per symmetry class, so e.g. a puzzle whose four solutions are
        rotations of one another reports one. Strictly less work than
        `symmetry` alone, since no image is ever built. It needs the
        group, so it uses the balanced path even under
        `branching="cell"`.
        """
        if branching not in ("cell", "block", "balanced"):
            raise ValueError(
                f"branching must be 'cell', 'block' or 'balanced', got {branching!r}."
            )

        n_total = self._placement_cells_flat.shape[0]
        n_cells = self.state.setup.n_cells
        if seed is None:
            self._priority = np.arange(n_total)
            self._cell_priority = np.arange(n_cells)
        else:
            rng = np.random.default_rng(seed)
            self._priority = rng.permutation(n_total)
            self._cell_priority = rng.permutation(n_cells)

        # The root's symmetry group, computed once. Every group the search
        # uses deeper is a subgroup of this one (a representative's
        # stabilizer), so the geometry is never searched again -- which is
        # what keeps symmetry off the per-node cost entirely. None means
        # "don't look", so a solve that can't use symmetry never builds
        # Setup's symmetry tables or placement lookup at all.
        root_images = None
        if up_to_symmetry or (symmetry and branching != "cell"):
            self._placement_lookup = self.state.setup.placement_lookup
            images = self.state.setup.region_symmetries(self.state.grid == EMPTY)
            if len(images) > 1:
                root_images = images
        block_branching = branching == "block"

        deadline = time.perf_counter() + time_limit
        found = 0
        stopped = max_solutions <= 0

        def search(live, block_count, cell_count, open_cells, images):
            """`live`/`block_count`/`cell_count` are a valid node (see
            _root_node), already pruned against `open_cells`, and
            owned by this call. `images` is the node's symmetry group, or
            None once nothing is left to collapse."""
            nonlocal found, stopped
            if stopped or time.perf_counter() >= deadline:
                # Sticky, so every caller up the stack unwinds too (each
                # undoing its own placement on the way out). Nothing ever
                # breaks out of a `for ... in search(...)` loop, which
                # would abandon a generator mid-placement.
                stopped = True
                return

            if block_count.min() == _PLACED:
                # Counted *before* the yield: a generator doesn't resume
                # until its consumer asks for the next item, and
                # orbit_branch reads `stopped` in between to decide
                # whether it may still derive this solution's images.
                found += 1
                stopped = found >= max_solutions
                # A full State snapshot, independent of self.state (which
                # keeps getting mutated as the search backtracks further).
                yield self.state.copy()
                return

            if images is not None:
                yield from orbit_branch(live, block_count, cell_count, open_cells, images)
                return

            if block_branching:
                branch = self._block_branch_candidates(live, block_count, cell_count, open_cells)
                if branch is None:
                    return
                pos, gids = branch
                candidates = [(pos, int(gid)) for gid in gids]
            else:
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

                yield from search(*child, open_cells, None)

                self.state.remove_unchecked(block_idx)
                open_cells.flat[placement] = True

        def orbit_branch(live, block_count, cell_count, open_cells, images):
            """A node whose open region is symmetric: branch on a block --
            the one branch set a symmetry carries onto itself -- and only
            search one placement per orbit, deriving the rest.

            Correct because every solution places that block exactly once,
            so the branches partition the solutions, and a symmetry of the
            open region is a bijection from the completions of one
            branch onto those of its image. Nothing is missed and nothing
            is found twice."""
            nonlocal found, stopped
            branch = self._block_branch_candidates(live, block_count, cell_count, open_cells)
            if branch is None:
                return
            pos, gids = branch
            block_idx = self._block_ids[pos]

            for rep, others, stabilizer in self._orbits(pos, gids, images):
                placement_idx = int(rep - self._block_start[pos])
                placement = self.placement_cells[block_idx][placement_idx]

                child = self._place(live, block_count, cell_count, pos, rep)
                if child is None:
                    continue

                self.state.place_unchecked(block_idx, placement_idx)
                open_cells.flat[placement] = False

                # The child's group is what's left of this one once the
                # representative is down. Usually nothing, and then the
                # whole subtree is ordinary branching; when it isn't, the
                # reduction simply nests and the factors multiply.
                for solution in search(*child, open_cells,
                                       stabilizer if len(stabilizer) > 1 else None):
                    yield solution
                    if up_to_symmetry:
                        continue
                    for image in others:
                        if stopped:
                            break
                        # Counted before the yield, for the same reason as
                        # in search()'s base case: when orbit reductions
                        # nest, the enclosing orbit_branch inspects
                        # `stopped` while this one is suspended here.
                        found += 1
                        stopped = found >= max_solutions
                        yield self._transform(solution, image)

                self.state.remove_unchecked(block_idx)
                open_cells.flat[placement] = True
                if stopped:
                    return

        node = self._root_node()
        if node is not None:
            yield from search(*node, self.state.grid == EMPTY, root_images)