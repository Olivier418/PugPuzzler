from __future__ import annotations

import math
import time
import weakref
from typing import TYPE_CHECKING

import numpy as np

from constants import EMPTY, UNPLACED

if TYPE_CHECKING:
    # Only for the type hints below -- never imported at runtime, so this
    # doesn't create a circular import with state.py (which imports
    # Solver at the top of the file).
    from .setup import Setup
    from .state import State


# counts entry for an item the search no longer has to satisfy: a covered
# cell or a placed block. Large enough that such an item can never be
# mistaken for the scarcest one, and that `counts.min() == _COVERED` means
# "everything is satisfied", i.e. solved.
_COVERED = 1 << 40


class _Tables:
    """Every table the search reads that depends on the Setup alone, laid
    out flat: one row per placement, one column per *item* the exact cover
    has to satisfy.

    An item is either one of the board's cells (`0 .. n_cells - 1`,
    "covered exactly once") or one of the blocks (`n_cells + block
    position`, "placed exactly once"). They are the same kind of
    constraint, so keeping them in one array is what lets a single count
    array, a single decrement and a single argmin serve the dead-end check
    and the branching choice for both (see `Solver._branch_item`). Column
    `n_cols` is padding, for rows shorter than the widest block.

    A placement's *global id* is its row in `placement_rows` /
    `placement_items`; block position p owns ids
    `block_start[p]:block_start[p + 1]`, so a block-local placement index
    is `gid - block_start[p]`. Nothing ever reorders these rows: a seed
    only changes the priorities that break ties (see `Solver.solve`), so
    everything indexed by placement stays valid.

    Built once per Setup and shared by every Solver on it (see
    `_tables_for`) -- a whole book is one board and one block collection,
    and none of this depends on which cells a particular puzzle starts
    with.
    """

    def __init__(self, setup: "Setup"):
        # Setup._validate has already rejected any block with no placements,
        # so every run below is non-empty and kmax is well defined.
        placement_cells = setup.placement_cells
        self.n_cells = n_cells = setup.n_cells
        self.block_ids = list(placement_cells)
        sizes = [placement_cells[b].shape[0] for b in self.block_ids]
        self.block_start = np.cumsum([0] + sizes, dtype=np.intp)
        self.block_of = np.repeat(np.arange(len(self.block_ids)), sizes)
        n_total = int(self.block_start[-1])
        self.n_cols = n_cols = n_cells + len(self.block_ids)

        # Short rows are padded with the padding column, which is parked at
        # _COVERED for the whole search: never scarce, never dead, and it
        # sorts last in _order_gids.
        kmax = max(arr.shape[1] for arr in placement_cells.values())
        rows = np.full((n_total, kmax), n_cols, dtype=np.intp)
        for pos, block_idx in enumerate(self.block_ids):
            arr = placement_cells[block_idx]
            rows[self.block_start[pos]:self.block_start[pos + 1], :arr.shape[1]] = arr
        self.placement_rows = rows
        self.placement_items = np.column_stack([rows, n_cells + self.block_of])

        # item -> global ids of every placement satisfying it. For a cell,
        # the placements through it: pair every real entry of `rows` with
        # its own gid, sort those pairs by cell, and cut the result at the
        # per-cell counts. For a block, simply its own contiguous run.
        flat = rows.ravel()
        real = flat < n_cells
        cell_of = flat[real]
        gids = np.repeat(np.arange(n_total), kmax)[real]
        by_cell = gids[np.argsort(cell_of, kind="stable")]
        bounds = np.cumsum(np.bincount(cell_of, minlength=n_cells))
        self.item_lists = np.split(by_cell, bounds[:-1])
        self.item_lists += [
            np.arange(self.block_start[p], self.block_start[p + 1])
            for p in range(len(self.block_ids))
        ]

        self._kill_cache = [None] * n_total

    def kill(self, gid) -> np.ndarray:
        """Global ids of every placement that placing `gid` takes out of
        the pool: everything sharing a cell with it, plus the rest of its
        own block (and itself), each listed once.

        Pure geometry, so computed on first request and kept -- a search
        revisits the same placements over and over, and every puzzle
        sharing this Setup reuses the same answer."""
        ids = self._kill_cache[gid]
        if ids is None:
            pos = self.block_of[gid]
            row = self.placement_rows[gid]
            own = np.arange(self.block_start[pos], self.block_start[pos + 1])
            ids = np.unique(np.concatenate(
                [self.item_lists[c] for c in row[row < self.n_cells]] + [own]
            ))
            self._kill_cache[gid] = ids
        return ids


# Setup -> its tables, so a book's puzzles build them once instead of once
# each. Worth 1.47x on main_puzzles and 2.09x on pyramid_puzzles over a whole
# book -- and almost none of that is the build itself (~0.03s across 72
# puzzles). It is `kill`'s memo: rebuilt per puzzle it starts cold every time,
# shared it is warm after the first few.
#
# Deliberately a module-level cache rather than an attribute on the Setup:
# benchmark.py pickles a Puzzle to each trial subprocess, and anything living
# in the Setup's __dict__ rides along. A warm cache is ~7.8 MB, which would
# take that pickle from 0.18 MB to 8.01 MB per trial.
#
# Weakly keyed, so nothing in here keeps a Setup (or its tables) alive.
_TABLES: "weakref.WeakKeyDictionary[Setup, _Tables]" = weakref.WeakKeyDictionary()


def _tables_for(setup: "Setup") -> _Tables:
    tables = _TABLES.get(setup)
    if tables is None:
        tables = _TABLES[setup] = _Tables(setup)
    return tables


class Solver:
    """Exhaustive exact-cover solver for a State (or Puzzle): every block
    must be placed and every open cell covered exactly once.

    Both halves of that are *items* to satisfy, counted in one array (see
    _Tables). The search branches on the scarcest item -- the cell or
    block that the fewest live placements can still satisfy -- and tries
    every placement satisfying it. Every solution satisfies that item
    exactly once, so those branches partition the solutions: nothing is
    missed and nothing is found twice.

    Where the still-open region is symmetric it branches on a block
    specifically, and searches only one placement per symmetry orbit; see
    solve()'s `symmetry`/`up_to_symmetry`. That only happens near the root
    -- the group dies as soon as a placement breaks it -- so the
    scarcest-item branch above is what runs essentially everywhere.

    Only reads State's public surface (board, blocks, placements,
    chosen_placement_idx, grid, place(), remove(), copy()) -- no
    knowledge of Puzzle's letter-grid loading or any other
    subclass-specific behavior is required.
    """

    def __init__(self, state: "State"):
        # A private working copy: it stays at the puzzle's starting
        # position (the search carries its own path and only materialises
        # a State when it reaches a solution), and there's no reason the
        # caller's own state should be tied to any of it. state.copy() is
        # cheap -- setup (board/blocks/placements) is shared by reference,
        # only the small per-instance grid + chosen_placement_idx dict are
        # copied.
        self.state = state.copy()

        # Blocks the state already considers placed (e.g. letters baked
        # into a Puzzle's starting grid) are fixed and excluded from the
        # search entirely.
        self.preplaced = {
            idx for idx, placement_idx in self.state.chosen_placement_idx.items()
            if placement_idx != UNPLACED
        }

        # The search relies on the unplaced blocks filling the open cells
        # exactly (that is what makes an item with no placement left a dead
        # end, and what lets it branch on any one item). Setup._validate
        # guarantees it: block cells == board cells, and preplaced blocks
        # only ever fill whole placements.

        # Shared with every other Solver on the same Setup, and unpacked
        # here so the search doesn't chase two attribute lookups per use.
        tables = _tables_for(self.state.setup)
        self._block_ids = tables.block_ids
        self._block_start = tables.block_start
        self._block_of = tables.block_of
        self._n_cells = tables.n_cells
        self._n_cols = tables.n_cols
        self._placement_rows = tables.placement_rows
        self._placement_items = tables.placement_items
        self._item_lists = tables.item_lists
        self._kill = tables.kill

        # Set per solve() call, declared here so the whole attribute
        # surface is in one place: the tie-break priorities drawn from the
        # seed, and Setup.placement_lookup, which is only fetched when a
        # solve can actually use symmetry so an unsymmetric one never
        # builds it.
        self._priority = None
        self._item_priority = None
        self._placement_lookup = None

    def _argmin_item(self, counts, lo, hi) -> int:
        """The scarcest item in `counts[lo:hi]`, as an item index. Ties go
        to the per-solve priorities `solve` draws from the seed, or to the
        lowest index when unseeded."""
        keys = counts[lo:hi]
        if self._item_priority is not None:
            keys = keys * self._n_cols + self._item_priority[lo:hi]
        return lo + int(keys.argmin())

    def _branch_item(self, counts) -> int | None:
        """The scarcest item, which is what this node branches on -- or -1
        if every item is already satisfied (a solution), or None if some
        item has no live placement left (a dead end -- nothing can ever
        satisfy it).

        One argmin over cells and blocks together answers all three: cells
        and blocks share the count array, a satisfied item holds _COVERED
        rather than 0, and both kinds of branch place exactly one block, so
        their counts compare directly."""
        item = self._argmin_item(counts, 0, self._n_cols)
        count = counts[item]
        if count == 0:
            return None
        if count == _COVERED:
            return -1
        return item

    def _root_node(self):
        """The search's starting node: every placement of a block that
        still has to be placed and fits the cells that are still empty, as
        `(live, counts, item)`, or None if the puzzle is dead on arrival.

        `live` is a bool array over global ids. `counts[i]` is how many
        live placements can still satisfy item i, or _COVERED for a cell a
        preplaced block already fills / a block already placed. `item` is
        what this node branches on (see _branch_item). Every later node is
        derived from its parent by _place, never rebuilt."""
        n_cells, n_cols = self._n_cells, self._n_cols
        unplaced = np.array([idx not in self.preplaced for idx in self._block_ids], dtype=bool)
        open_cells = self.state.grid == EMPTY
        filled = np.zeros(n_cols + 1, dtype=bool)  # padding column never blocks
        filled[:n_cells] = ~open_cells

        live = unplaced[self._block_of] & ~filled[self._placement_rows].any(axis=1)
        counts = np.bincount(self._placement_items[live].ravel(), minlength=n_cols + 1)
        counts[:n_cells][~open_cells] = _COVERED
        counts[n_cells:n_cols][~unplaced] = _COVERED
        counts[n_cols] = _COVERED  # the padding column, parked once and for all

        item = self._branch_item(counts)
        if item is None:
            return None
        return live, counts, item

    def _place(self, live, counts, gid):
        """The child node after placing the placement with global id `gid`:
        every placement it conflicts with leaves the pool (its own block's
        included -- see _Tables.kill), and the items it satisfies are
        marked _COVERED. Everything is derived from the parent's arrays by
        touching only the placements that actually change, rather than
        re-testing anything.

        The child's branching item is chosen here too, so a child that is
        already a dead end is rejected before the search ever descends into
        it -- about a third of all children, each of which would otherwise
        cost a recursion and a full node's bookkeeping first.

        The parent's arrays are left alone (siblings share them); returns
        `(live, counts, item)` for the child, or None if it is dead."""
        kill = self._kill(gid)
        hit = kill[live[kill]]

        new_counts = counts - np.bincount(
            self._placement_items[hit].ravel(), minlength=counts.size
        )
        new_counts[self._placement_items[gid]] = _COVERED

        item = self._branch_item(new_counts)
        if item is None:
            return None

        # Only worth copying once the child is known to survive.
        new_live = live.copy()
        new_live[hit] = False
        return new_live, new_counts, item

    def _order_gids(self, gids, counts):
        """`gids` in visit order: lexicographic by each placement's cells'
        live placement counts, sorted ascending and compared like tuples
        -- so a placement covering the scarcest cells comes first, with no
        weights to tune. Whatever is still tied (equal count vectors) is
        decided by the per-solve priorities `solve` draws from the seed
        (identity order when unseeded).

        Shared by the item branch and the symmetry orbit sweep, so both
        agree on which candidate is "best" -- which is what makes the orbit
        representative the best-ranked member of its orbit rather than an
        arbitrary one. Only pays for itself when the search is after the
        first few solutions rather than all of them; see solve()'s
        `order`."""
        keys = counts[self._placement_rows[gids]]
        keys.sort(axis=1)  # padding is parked at _COVERED, so it sorts last
        # lexsort treats its *last* key as primary.
        return gids[np.lexsort((self._priority[gids], *keys.T[::-1]))]

    def _block_branch(self, live, counts, order):
        """The still-unplaced block with the fewest live placements, as
        `(block position, global ids in visit order)`.

        Every solution places that block exactly once, so these branches
        partition the solutions -- and unlike a cell's placements, the set
        of them is carried onto itself by any symmetry of the open region,
        which is what lets `solve` collapse it into orbits (see _orbits).
        That is its only purpose: `orbit_branch` is the sole caller, since
        everywhere else _branch_item's scarcest item is the better branch
        whichever kind it turns out to be."""
        item = self._argmin_item(counts, self._n_cells, self._n_cols)
        gids = self._item_lists[item]
        gids = gids[live[gids]]
        if order:
            gids = self._order_gids(gids, counts)
        return item - self._n_cells, gids

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
        cells = self.state.placement_cells[block_idx][placement_idx]
        return self._placement_lookup[block_idx][np.sort(image[cells]).tobytes()]

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
        # This block's gids are the contiguous run starting at `lo`, so a
        # gid and its block-local placement index differ by that constant.
        lo = int(self._block_start[pos])
        block_idx = self._block_ids[pos]

        visited = set()
        for gid in gids.tolist():
            if gid in visited:
                continue
            orbit = {}
            stabilizer = []
            for image in images:
                other = lo + self._image_placement_idx(block_idx, gid - lo, image)
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
        symmetry: bool = True,
        up_to_symmetry: bool = False,
        order: bool = None,
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
        per item, used only to break ties the search's own ordering
        leaves open. Unseeded, ties fall to placement/item index order.
        Nothing is ever reordered or shared, so a seed can't disturb
        anything indexed by placement.

        Every node branches on the scarcest item of either kind (see
        _branch_item) -- the cell covered, or the block placeable, by the
        fewest live placements -- and tries every placement satisfying it.
        Branching on a block the moment it is scarcer than every cell is
        what a plain cell rule misses, and it costs nothing: the counts
        share one array. There is no mode to pick; the cell-only,
        block-only and hybrid rules this replaced were measured and
        removed (SYMMETRY_NOTES.md section 6).

        `symmetry` turns the orbit reduction on: at a node whose open
        region has a nontrivial symmetry group, the search branches on a
        block (the one branch set a symmetry carries onto itself), groups
        its placements into orbits, searches one representative per orbit,
        and produces the other members' solutions by transforming the
        representative's. The solution *set* is unchanged; the order is
        not (a solution is followed by its images). Where the open region
        is asymmetric it costs nothing beyond the one group lookup.

        `up_to_symmetry` yields only the representatives -- one solution
        per symmetry class, so e.g. a puzzle whose four solutions are
        rotations of one another reports one. Strictly less work than
        `symmetry` alone, since no image is ever built. It needs the
        group, so it looks one up whether or not `symmetry` is set.

        `order` sorts each node's candidate placements, scarcest cells
        first (see _order_gids). It changes the order solutions come out
        in, never the set. Left as None it is on exactly when a limit is
        set, which is when it can pay: it finds the first solutions
        markedly sooner, but a full enumeration visits the same nodes
        whatever order siblings are tried in, so there it is pure
        overhead (~25% of the runtime).
        """
        n_cols = self._n_cols
        n_total = self._placement_items.shape[0]
        if seed is None:
            self._priority = np.arange(n_total)
            self._item_priority = None  # argmin already breaks ties by index
        else:
            rng = np.random.default_rng(seed)
            self._priority = rng.permutation(n_total)
            self._item_priority = rng.permutation(n_cols)

        if order is None:
            order = time_limit < math.inf or max_solutions < math.inf

        # The root's symmetry group, computed once. Every group the search
        # uses deeper is a subgroup of this one (a representative's
        # stabilizer), so the geometry is never searched again -- which is
        # what keeps symmetry off the per-node cost entirely. None means
        # "don't look", so a solve that turns symmetry off never builds
        # Setup's symmetry tables or placement lookup at all.
        root_images = None
        if up_to_symmetry or symmetry:
            self._placement_lookup = self.state.setup.placement_lookup
            images = self.state.setup.region_symmetries(self.state.grid == EMPTY)
            if len(images) > 1:
                root_images = images

        deadline = time.perf_counter() + time_limit
        found = 0
        stopped = max_solutions <= 0

        # The placements chosen down to the current node, as global ids.
        # The search itself never touches a grid; a node that turns out to
        # be a solution replays this onto a copy of the starting state.
        path = []
        item_lists = self._item_lists
        place = self._place

        def snapshot():
            """The current path as a full State, independent of
            self.state (which stays at the starting position)."""
            solution = self.state.copy()
            for gid in path:
                pos = self._block_of[gid]
                solution.place_unchecked(
                    self._block_ids[pos], int(gid - self._block_start[pos])
                )
            return solution

        def search(live, counts, item, images):
            """`live`/`counts` are a valid node (see _root_node) owned by
            this call, and `item` is what it branches on, or -1 if it is a
            solution. `images` is the node's symmetry group, or None once
            nothing is left to collapse."""
            nonlocal found, stopped
            if stopped or time.perf_counter() >= deadline:
                # Sticky, so every caller up the stack unwinds too (each
                # popping its own placement on the way out). Nothing ever
                # breaks out of a `for ... in search(...)` loop, which
                # would abandon a generator mid-placement.
                stopped = True
                return

            if item == -1:
                # Counted *before* the yield: a generator doesn't resume
                # until its consumer asks for the next item, and
                # orbit_branch reads `stopped` in between to decide
                # whether it may still derive this solution's images.
                found += 1
                stopped = found >= max_solutions
                yield snapshot()
                return

            if images is not None:
                yield from orbit_branch(live, counts, images)
                return

            gids = item_lists[item]
            gids = gids[live[gids]]
            if order:
                gids = self._order_gids(gids, counts)

            for gid in gids.tolist():
                child = place(live, counts, gid)
                if child is None:
                    continue

                path.append(gid)
                yield from search(*child, None)
                path.pop()

        def orbit_branch(live, counts, images):
            """A node whose open region is symmetric: branch on a block --
            the one branch set a symmetry carries onto itself -- and only
            search one placement per orbit, deriving the rest.

            Correct because every solution places that block exactly once,
            so the branches partition the solutions, and a symmetry of the
            open region is a bijection from the completions of one
            branch onto those of its image. Nothing is missed and nothing
            is found twice."""
            nonlocal found, stopped
            pos, gids = self._block_branch(live, counts, order)

            for rep, others, stabilizer in self._orbits(pos, gids, images):
                child = place(live, counts, rep)
                if child is None:
                    continue

                path.append(rep)

                # The child's group is what's left of this one once the
                # representative is down. Usually nothing, and then the
                # whole subtree is ordinary branching; when it isn't, the
                # reduction simply nests and the factors multiply.
                for solution in search(*child, stabilizer if len(stabilizer) > 1 else None):
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

                path.pop()
                if stopped:
                    return

        node = self._root_node()
        if node is not None:
            yield from search(*node, root_images)
