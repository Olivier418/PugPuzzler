"""The Python side of the search: solve_rows drives the compiled kernel
(classes.kernel) over a Puzzle and yields its solutions as rows. It owns
what the kernel cannot see -- the Puzzle, the options, the clock and the
lazy generator contract -- and is otherwise thin."""
import math
import time

import numpy as np

from pugpuzzler.constants import EMPTY, UNPLACED
from . import kernel


# branch= accepts these names; see solve_rows.
_BRANCH_RULES = {
    "both": kernel.BRANCH_BOTH,
    "cell": kernel.BRANCH_CELL,
    "block": kernel.BRANCH_BLOCK,
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


def _resolve_rules(seed, time_limit, max_solutions, order, branch):
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
        # Unranked, a seed would do nothing (the branch argmin never sees
        # it), so a seeded run sorts by its drawn priority alone.
        rank_rule = kernel.RANK_PRIORITY if seed is not None else kernel.RANK_NONE
    else:
        raise ValueError(f"order={order!r}; expected True, False or None.")

    return branch_rule, rank_rule


def _start(puzzle, t: kernel.Tables):
    """The puzzle's starting position as the kernel sees it: `(occ, used)`.
    Pre-placed blocks (the letters of a Puzzle's starting grid) are fixed,
    and excluded from the search by marking them used."""
    occ = np.zeros(t.full.size, dtype=np.uint64)
    for c in np.flatnonzero(puzzle.grid != EMPTY):
        occ[c >> 6] |= kernel.BIT[c & 63]

    used = np.uint64(0)
    for idx, placement_idx in puzzle.chosen_placement_idx.items():
        if placement_idx != UNPLACED:
            used |= kernel.BIT[idx]
    return occ, used


def solve_rows(
    puzzle,
    seed: int | None = None,
    time_limit: float = math.inf,
    max_solutions: float = math.inf,
    order: bool | None = None,
    branch: str = "both",
):
    """Yield every solution of `puzzle` (exhaustive exact cover: every block
    placed, every open cell covered exactly once) as a row: one int64 array
    holding the placement index of every block -- pre-placed ones included
    -- in block order. Setup.rows_to_grids turns rows into grids,
    Puzzle.solved_copy into a Puzzle.

    Stops early once a limit is hit. The generator's return value
    (what `yield from` evaluates to, or StopIteration.value) says
    whether it did: True if the search ran to the end, so every
    solution was yielded; False if a limit cut it short. This is what
    becomes SolveStats.complete.

    `time_limit` (seconds, counted from the first `next()`) and
    `max_solutions` both default to infinity; the search stops as soon
    as either is reached. The kernel runs in ~2 ms chunks (sized by an
    EMA of nodes/s), so `time_limit` is exact to ~2 ms and the generator
    can be abandoned mid-tree.

    `seed` randomises the order solutions are found in, never the set:
    one random priority per placement breaks the ties the ranking leaves.

    Every node branches on one *item* and tries every placement that
    settles it, so the branches partition the node's solutions. `branch`
    picks which kind of item, and only affects speed:

    - "both" (the default): whichever of the two below has fewer live
      placements at this node; wins or ties everywhere.
    - "cell": always the empty cell with the fewest live placements.
    - "block": always the unplaced block with the fewest. A baseline:
      it can't see a cell nothing covers until some block runs out of
      room, so it barely prunes an open board (the empty main board: no
      solution in 30 s, against ~1 ms for "both").

    Measurements for the three modes are in docs/DESIGN_NOTES.md.

    `order` ranks each node's candidate placements, in all three `branch`
    modes; it changes the order solutions come out in, never the set.

    - True: by each candidate's cells' live-placement counts, sorted
      ascending and compared like a tuple, so a placement covering the
      scarcest cells goes first.
    - False: table order (or, seeded, by the seed's priority alone).
    - None (the default): True exactly when a limit is set -- a full
      enumeration visits the same nodes in any order, so there it is
      pure overhead.
    """
    branch_rule, rank_rule = _resolve_rules(seed, time_limit, max_solutions, order, branch)
    if max_solutions <= 0 or time_limit <= 0:
        return False

    t = puzzle.setup.kernel_tables
    n_placements = t.pmask.shape[0]
    if seed is None:
        priority = np.arange(n_placements, dtype=np.int32)
    else:
        priority = np.random.default_rng(seed).permutation(n_placements).astype(np.int32)

    # Every solution row starts as this one: the pre-placed blocks'
    # columns never change, and a searched path fills in the rest.
    base = np.array(list(puzzle.chosen_placement_idx.values()), dtype=np.int64)

    # The kernel is never told max_solutions: it gets a per-call buffer cap
    # and a node budget, returns FULL/BUDGET/DONE, and the counting stays here.
    cap = _SOL_CAP
    if max_solutions < math.inf:
        cap = int(min(_SOL_CAP, math.ceil(max_solutions)))
    w = kernel.make_workspace(t, *_start(puzzle, t), cap=cap)

    deadline = time.perf_counter() + time_limit
    found = 0
    rate = None  # nodes per second, an EMA; only sizes chunks, never affects output
    while True:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return False
        if rate is None:
            budget = _CALIB_CHUNK
        else:
            budget = int(min(max(rate * min(_CHUNK_SECONDS, remaining), _MIN_CHUNK), _MAX_CHUNK))
        t0 = time.perf_counter()
        status, n_sol, nodes = kernel.kernel(t, w, cap, budget, rank_rule, priority, branch_rule)
        elapsed = time.perf_counter() - t0
        if nodes > 0 and elapsed > 0:
            rate = nodes / elapsed if rate is None else 0.7 * rate + 0.3 * nodes / elapsed

        if n_sol:
            take = int(min(n_sol, max_solutions - found))
            found += take
            # The kernel's paths are global placement ids, (take, n_searched);
            # a row holds each block's own placement index at its column.
            # Built as a fresh array, so re-entering the kernel (which
            # reuses sol_buf) can't touch rows already yielded.
            paths = w.sol_buf[:take, :w.sol_len[0]]
            pos = t.pblock[paths]
            rows = np.repeat(base[None, :], take, axis=0)
            np.put_along_axis(rows, pos, paths - t.block_start[pos], axis=1)
            yield from rows
            if take < n_sol:  # max_solutions cut this batch short
                return False
        if status == kernel.DONE:
            return True
        if found >= max_solutions:
            return False
