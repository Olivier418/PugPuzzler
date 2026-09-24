from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING

import numpy as np

from constants import EMPTY, UNPLACED
from . import kernel

if TYPE_CHECKING:
    # Only for the type hints below -- never imported at runtime, so this
    # doesn't create a circular import with state.py (which imports
    # Solver at the top of the file).
    from .state import State


# branch= accepts these names; see Solver.solve.
_BRANCH_RULES = {
    "both": kernel.BRANCH_BOTH,
    "cell": kernel.BRANCH_CELL,
    "block": kernel.BRANCH_BLOCK,
}

# order= accepts these names as well as True/False; see Solver.solve.
_RANK_RULES = {
    "counts": kernel.RANK_COUNTS,
    "pockets": kernel.RANK_POCKETS,
    "fanout": kernel.RANK_FANOUT,
}

# Kernel chunking. The kernel runs until it has `cap` solutions or has
# spent `budget` nodes, then hands control back so the driver can check
# the clock and yield. 2 ms against a ~5 us njit dispatch is ~0.25%
# overhead, and bounds how far a time_limit can overshoot.
_CHUNK_SECONDS = 0.002
_MIN_CHUNK = 2048
_MAX_CHUNK = 1 << 22
_CALIB_CHUNK = 4096
_SOL_CAP = 64


class Solver:
    """Exhaustive exact-cover solver for a State (or Puzzle): every block
    must be placed and every open cell covered exactly once.

    The search itself lives in classes.kernel, compiled by numba: the
    board's occupancy is a bitmask, so "does this placement fit" is one
    AND, and a node costs a few hundred integer operations rather than a
    dozen NumPy calls over conflict lists. This class owns everything the
    kernel cannot -- the Setup/State objects and the lazy generator
    contract -- and is otherwise thin: it sets up the starting position,
    then feeds the kernel node budgets until it is done.

    **Laziness.** The kernel is never told `max_solutions`; it gets a
    solution-buffer cap and a node budget, returns FULL/BUDGET/DONE, and
    is re-entered with its stack intact. An EMA of nodes/s sizes ~2 ms
    chunks, which is how `time_limit` stays exact to ~2 ms and how the
    generator stays abandonable mid-tree.

    Only reads State's public surface (setup, grid, chosen_placement_idx,
    placement_cells, place_unchecked(), copy()) -- no knowledge of
    Puzzle's letter-grid loading or any other subclass-specific
    behaviour is required.
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

        # Built once per Setup and shared by every Solver on it -- a whole
        # book is one board and one block collection. See
        # Setup.kernel_tables.
        #
        # The search relies on the unplaced blocks filling the open cells
        # exactly: it calls a node solved the moment every cell is
        # covered, without separately checking that every block went
        # down. Setup._validate guarantees the equivalence -- block cells
        # == board cells, and preplaced blocks only ever fill whole
        # placements.
        self.tables, self.block_ids = self.state.setup.kernel_tables

        # Set per solve() call, declared here so the whole attribute
        # surface is in one place: the tie-break priorities drawn from the
        # seed.
        self._priority = None

        # Nodes per second, an EMA used to size kernel chunks. It
        # survives across solve() calls on purpose: a chunk boundary
        # changes only *where the kernel returns*, never the order in
        # which it visits nodes, so carrying it cannot affect output.
        self._rate = None

    # ---- chunk sizing ---------------------------------------------------

    def _next_budget(self, remaining: float) -> int:
        if self._rate is None:
            return _CALIB_CHUNK
        want = self._rate * min(_CHUNK_SECONDS, max(remaining, 0.0))
        return int(min(max(want, _MIN_CHUNK), _MAX_CHUNK))

    def _observe(self, nodes: int, elapsed: float):
        if elapsed <= 0 or nodes <= 0:
            return
        rate = nodes / elapsed
        self._rate = rate if self._rate is None else 0.7 * self._rate + 0.3 * rate

    # ---- the starting position ------------------------------------------

    def _start(self):
        """The puzzle's starting position as the kernel sees it:
        `(occ, used)`. Blocks the state already considers placed (e.g.
        letters baked into a Puzzle's starting grid) are fixed, and are
        excluded from the search by marking them used."""
        occ = np.zeros(self.tables.full.size, dtype=np.uint64)
        for c in np.flatnonzero(self.state.grid != EMPTY):
            occ[c >> 6] |= kernel.BIT[c & 63]

        used = np.uint64(0)
        chosen = self.state.chosen_placement_idx
        for pos, block_idx in enumerate(self.block_ids):
            if chosen[block_idx] != UNPLACED:
                used |= kernel.BIT[pos]
        return occ, used

    def _emit(self, path):
        """A path of global ids as a full State, independent of
        self.state (which stays at the starting position).

        place_unchecked writes the grid and chosen_placement_idx
        together, so a yielded solution can never have one without the
        other -- which is what tests/_helpers.assert_valid_solution
        cross-checks."""
        t = self.tables
        solution = self.state.copy()
        for gid in path:
            pos = int(t.pblock[gid])
            solution.place_unchecked(
                self.block_ids[pos], int(gid - t.block_start[pos])
            )
        return solution

    def _rules(self, seed, time_limit, max_solutions, order, branch):
        """Validate and resolve the `branch` / `order` options into the
        integer rule codes the kernel takes."""
        try:
            branch_rule = _BRANCH_RULES[branch]
        except (KeyError, TypeError):
            raise ValueError(
                f"branch={branch!r}; expected one of {sorted(_BRANCH_RULES)}."
            ) from None

        if order is None:
            order = time_limit < math.inf or max_solutions < math.inf

        if order is True:
            rank_rule = kernel.RANK_COUNTS
        elif order is False:
            # With no ranking at all a seed would be a no-op: the kernel
            # picks the branch item by an argmin the seed never touches.
            # Sorting by the drawn priority alone keeps the documented
            # promise that a seed randomises arrival order, at almost no
            # cost.
            rank_rule = kernel.RANK_PRIORITY if seed is not None else kernel.RANK_NONE
        else:
            try:
                rank_rule = _RANK_RULES[order]
            except (KeyError, TypeError):
                raise ValueError(
                    f"order={order!r}; expected True, False, or one of "
                    f"{sorted(_RANK_RULES)}."
                ) from None

        return branch_rule, rank_rule

    def solve(
        self,
        seed: int = None,
        time_limit: float = math.inf,
        max_solutions: float = math.inf,
        order=None,
        branch: str = "both",
    ):
        """Yield every solution as a fully placed copy of the state, or --
        if a limit is hit first -- just the ones found by then.

        `time_limit` (seconds, counted from the first `next()`) and
        `max_solutions` both default to infinity; the search stops as soon
        as either is reached. The time is checked at kernel chunk
        boundaries, so it can overshoot by about one chunk (~2 ms).

        `seed` randomises the order solutions are found in without
        changing the set: it draws one random priority per placement,
        used only to break ties the search's own ordering leaves open.
        Unseeded, ties fall to placement index order. Nothing is ever
        reordered or shared, so a seed can't disturb anything indexed by
        placement.

        Every node branches on one *item* and tries every placement that
        settles it, so the branches partition the node's solutions:
        nothing is missed and nothing is found twice. `branch` picks which
        kind of item, and only affects speed:

        - "both" (the default): whichever of the two below has fewer live
          placements at this node. The counts are directly comparable --
          a child of either kind places exactly one block -- so this is
          simply better informed than either alone, and it wins or ties
          everywhere.
        - "cell": always the empty cell with the fewest live placements.
        - "block": always the unplaced block with the fewest. Sound, and
          a useful baseline, but much the weakest: it cannot notice a
          cell that nothing can cover until some block runs out of room,
          so it prunes almost nothing where little is pre-filled. On a
          book puzzle that costs a factor (worst measured:
          main_puzzles/65, 2 s against 0.01 s); on the empty main board
          it finds no solution at all in 30 s, against ~1 ms for "both".

        Full enumeration of each book, and a fixed solution count on the
        empty boards (time in seconds, then nodes):

            IQpuzzler main_puzzles    both 0.55 / 204833   cell 0.62 /  260916
            IQpuzzler pyramid_puzzles both 0.04 /  10568   cell 0.11 /   36916
            PRO main_puzzles          both 0.07 /  21015   cell 0.07 /   21767
            PRO pyramid_puzzles       both 0.69 / 233264   cell 0.75 /  264095
            empty_main, 2000 sols     both 0.49 / 204692   cell 1.67 /  685857
            empty_pyramid, 500 sols   both 1.59 / 548864   cell 14.35 / 4853760

        `order` ranks each node's candidate placements. It changes the
        order solutions come out in, never the set. The same rule is
        applied to whatever candidate list the node built, and every rule
        keys off a placement rather than off the item that was branched
        on, so ordering is uniform across all three `branch` modes by
        construction. Left as None it is on exactly when a limit is set,
        which is when it can pay: it finds the first solutions markedly
        sooner, but a full enumeration visits the same nodes whatever
        order siblings are tried in, so there it is pure overhead.
        Accepts:

        - True / "counts": each candidate's cells' live-placement counts,
          sorted ascending and compared like a tuple, so a placement
          covering the scarcest cells goes first.
        - "pockets": fewest empty cells stranded with no empty neighbour.
          Those cells can never be covered, so this rejects a dead branch
          one ply early.
        - "fanout": fewest options left for the next empty cell.
        - False: table order (but see the seed note above).
        """
        t = self.tables
        if seed is None:
            self._priority = np.arange(t.pmask.shape[0], dtype=np.int32)
        else:
            rng = np.random.default_rng(seed)
            self._priority = rng.permutation(t.pmask.shape[0]).astype(np.int32)

        branch_rule, rank_rule = self._rules(
            seed, time_limit, max_solutions, order, branch
        )

        if max_solutions <= 0 or time_limit <= 0:
            return

        deadline = time.perf_counter() + time_limit
        found = 0

        # The kernel is never told max_solutions -- it gets a per-call
        # buffer cap and a node budget, and all the counting stays here.
        cap = _SOL_CAP
        if max_solutions < math.inf:
            cap = int(min(_SOL_CAP, math.ceil(max_solutions)))
        w = kernel.make_workspace(t, *self._start(), cap=cap)

        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                return
            budget = self._next_budget(remaining)
            t0 = time.perf_counter()
            status, n_sol, nodes = kernel.kernel(
                t, w, cap, budget, rank_rule, self._priority, branch_rule
            )
            self._observe(nodes, time.perf_counter() - t0)

            for i in range(n_sol):
                found += 1
                yield self._emit(w.sol_buf[i, :w.sol_len[i]].tolist())
                if found >= max_solutions:
                    return
            if status == kernel.DONE:
                return
