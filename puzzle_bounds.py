"""The solution bounds of a board: for every attainable number of filled
cells, the fewest solutions (the lower bound, the hardest puzzle) and the most
(the upper bound, the easiest) any puzzle with that many filled cells has --
exactly, from the empty board (0 filled, every solution) out to one block
short of a full board, with one example puzzle per bound per filled count
(also where the bound is 1).

Since Setup._validate_area guarantees the blocks' total area equals the
board's, a puzzle's filled count depends only on *which* blocks are
pre-placed, so the search is over subsets of blocks. Each subset's
*realizations* are its placements that leave the board solvable. Pre-placing
blocks never creates a solution, it only rules them out: a realization's
solutions are exactly the empty board's solutions that place those blocks
the same way. So with every empty-board solution as a row of placement
indices (Solution.rows of a complete solve of the empty puzzle), a
subset's realizations and their solution counts are the groups the rows
fall into by that subset's columns -- no solver call needed. Nothing is
sampled or estimated.

bound_counts does that grouping for every subset: a DFS that grows a
subset one (higher) block at a time, each child's groups refining its
parent's. Two facts keep it cheap:

  - A realization with a unique solution stays unique however many further
    blocks are placed the way that solution places them, so it settles the
    minimum of every superset at 1 at once.
  - A child's groups never outgrow its parent's. Once the minimum is settled
    below a node, only groups big enough to beat the maximum at some filled
    count still reachable from it are passed on; the rest of the rows are
    dropped for the whole subtree.

The bounds' puzzles are then solved once more, on solving.timed_solve's
clock, for their SolveStats (and each count is checked against the
grouping's). Nothing here reads or writes files: serialization.save_bounds /
load_bounds keep both books -- puzzles and stats -- in the board's bounds
folder, bounds/<game>/<board_key>/."""
import itertools
import math

import numpy as np
from tqdm import tqdm

from classes import Puzzle, PuzzleBook, Setup, Solution, SolveStats, SolveStatsBook
from constants import EASIEST_DIFFICULTY, HARDEST_DIFFICULTY, LOWER_BOUND_BOOK, UPPER_BOUND_BOOK
from serialization import Bounds
from solving import timed_solve

# filled count -> (nr_solutions, realization); a realization is
# {column: placement_idx}, a column being a block's position in the Setup's
# block order (the table's columns).
Bound = dict[int, tuple[int, dict[int, int]]]


class _Grouping:
    """bound_counts' DFS state. A subset is a sorted tuple of columns."""

    def __init__(self, table: np.ndarray, n_placements: list[int], sizes: list[int], verbose: bool):
        self.table = table
        self.n_placements = n_placements
        self.sizes = sizes
        self.n = table.shape[1]
        self.full = sum(sizes)
        self.lower: Bound = {0: (len(table), {})}
        self.upper: Bound = {0: (len(table), {})}
        # Every non-empty subset, the full board included, is accounted for
        # once: visited, skipped (full) or pruned with its parent's subtree.
        self.pbar = tqdm(total=2 ** self.n - 1, desc="Grouping block subsets", unit=" subsets",
                         disable=not verbose)

    def filled(self, cols) -> int:
        return sum(self.sizes[c] for c in cols)

    def reachable(self, filled: int, cols: list[int]) -> set[int]:
        """The filled counts of adding any non-empty subset of `cols` to a
        subset `filled` large, short of the full board."""
        reach = {filled}
        for c in cols:
            reach |= {x + self.sizes[c] for x in reach}
        reach.discard(filled)
        reach.discard(self.full)
        return reach

    def realization(self, cols, rows, inv, group) -> tuple[dict[int, int], np.ndarray]:
        """The realization of `cols` that `group` stands for, and one of its
        solution rows."""
        row = self.table[rows[np.argmax(inv == group)]]
        return {c: int(row[c]) for c in cols}, row

    def settle(self, cols: tuple[int, ...], row: np.ndarray):
        """`row` is the only solution placing `cols` its way: every superset
        of `cols` short of the full board, realized the way `row` places the
        extra blocks, has exactly one solution too."""
        rest = [c for c in range(self.n) if c not in cols]
        filled = self.filled(cols)
        if all(self.lower.get(f, (math.inf,))[0] == 1 for f in self.reachable(filled, rest)):
            return
        for r in range(1, len(rest)):  # all of `rest` would be the full board
            for extra in itertools.combinations(rest, r):
                f = filled + self.filled(extra)
                if self.lower.get(f, (math.inf,))[0] > 1:
                    self.lower[f] = (1, {c: int(row[c]) for c in sorted(cols + extra)})

    def dfs(self, cols: tuple[int, ...], gid: np.ndarray, rows: np.ndarray, settled: bool):
        """Every subset extending `cols` by higher columns. `rows` are the
        table rows still in play and `gid` their group under `cols`;
        `settled` means every such subset's minimum is already 1."""
        filled = self.filled(cols)
        for c in range(cols[-1] + 1 if cols else 0, self.n):
            sub = cols + (c,)
            self.pbar.update(1)
            if len(sub) == self.n:  # the finished board
                continue
            _, inv, cnt = np.unique(
                gid * self.n_placements[c] + self.table[rows, c], return_inverse=True, return_counts=True,
            )
            f = filled + self.sizes[c]

            hi = int(cnt.max())
            if hi > self.upper.get(f, (0,))[0]:
                self.upper[f] = (hi, self.realization(sub, rows, inv, int(cnt.argmax()))[0])
            if not settled:
                lo = int(cnt.min())
                improves = lo < self.lower.get(f, (math.inf,))[0]
                if improves or lo == 1:
                    realization, row = self.realization(sub, rows, inv, int(cnt.argmin()))
                    if improves:
                        self.lower[f] = (lo, realization)
                    if lo == 1:
                        self.settle(sub, row)

            later = list(range(c + 1, self.n))
            reach = self.reachable(f, later)
            subtree = 2 ** len(later) - 1
            if not reach:
                self.pbar.update(subtree)
                continue
            if settled or lo == 1:
                # Only the maximum is left to find below: a group no bigger
                # than every reachable count's best maximum can't beat any.
                keep = cnt[inv] > min(self.upper.get(x, (0,))[0] for x in reach)
                if not keep.any():
                    self.pbar.update(subtree)
                    continue
                self.dfs(sub, inv[keep], rows[keep], True)
            else:
                self.dfs(sub, inv, rows, False)


def bound_counts(
    table: np.ndarray, n_placements: list[int], sizes: list[int], verbose: bool = False
) -> tuple[Bound, Bound]:
    """(lower, upper): for every filled count reachable by pre-placing a
    subset of the blocks (0 included, the full board excluded), the fewest
    and the most solutions over all its realizations, each with one
    realization that has it. `table` holds every empty-board solution as
    a row of placement indices, one column per block; `n_placements` and
    `sizes` are each column's number of placements and cell count."""
    grouping = _Grouping(np.asarray(table), n_placements, sizes, verbose)
    n_rows = len(grouping.table)
    grouping.dfs((), np.zeros(n_rows, dtype=np.int64), np.arange(n_rows), False)
    grouping.pbar.close()
    return grouping.lower, grouping.upper


def _bound_puzzles(setup: Setup, bound: Bound, name: str, difficulty: str) -> PuzzleBook:
    """The puzzles of `bound`, named 1, 2, 3, ... by increasing filled count
    and tagged `difficulty`."""
    block_idxs = list(setup.blocks)
    puzzles = []
    for i, filled in enumerate(sorted(bound), start=1):
        puzzle = Puzzle(setup, name=str(i), difficulty=difficulty)
        for col, placement_idx in bound[filled][1].items():
            puzzle.place_unchecked(block_idxs[col], placement_idx)
        puzzles.append(puzzle)
    return PuzzleBook(*puzzles, name=name)


def _time_bound_book(
    puzzles: PuzzleBook, empty_stats: SolveStats, verbose: bool, expected: list[int] | None = None
) -> SolveStatsBook:
    """The SolveStats of a bound book's puzzles: the empty board's (the
    first) is `empty_stats`, every other one is solved here, and checked
    against its `expected` solution count if given (it can only differ if
    the empty board's run isn't truly complete)."""
    empty, *rest = puzzles.values()
    stats = [empty_stats._replace(puzzle_name=empty.name)]
    for i, puzzle in enumerate(tqdm(rest, desc=f"Timing {puzzles.name}", disable=not verbose), start=1):
        _, puzzle_stats = timed_solve(puzzle, keep_rows=False)
        if expected is not None and len(puzzle_stats.elapsed) != expected[i]:
            raise RuntimeError(
                f"Puzzle {puzzle.name} of {puzzles.name} has {len(puzzle_stats.elapsed)} solutions, but the "
                f"empty board's run says {expected[i]}: is that run really complete?"
            )
        stats.append(puzzle_stats)
    return SolveStatsBook(*stats, book_name=puzzles.name, options={}, seed=None)


def compute_puzzle_bounds(empty_solution: Solution, empty_stats: SolveStats, verbose: bool = True) -> Bounds:
    """The bounds on the number of solutions at every attainable number of
    filled cells on a board, from every solution of its empty board: one
    puzzle with the fewest solutions possible at that count (the lower
    bound) and one with the most (the upper bound) -- exact (ties go to
    whichever the search meets first), one example per count even where the
    bound is 1. Both include the empty board (0 filled) and exclude the
    finished one.

    `empty_solution`/`empty_stats` are a complete solve of an empty puzzle
    (e.g. solving.empty_puzzle's), solved now with solving.solve_puzzle or
    loaded with serialization.load_puzzle_run -- ~15 min to solve for
    IQpuzzlerPRO main, so save it. The board is the one that puzzle is on.

    Returns ((lower_puzzles, lower_stats), (upper_puzzles, upper_stats)):
    two PuzzleBooks, named constants.LOWER_BOUND_BOOK / UPPER_BOUND_BOOK,
    with their puzzles named 1, 2, 3, ... by increasing filled count and
    tagged constants.HARDEST_DIFFICULTY / EASIEST_DIFFICULTY (colored by
    plotting.plot_puzzle_stats via constants.BOUND_COLORS), and each book's
    SolveStatsBook, puzzle 1's being `empty_stats`. plotting.plot_bounds
    draws them as a band. No SolutionBooks: the empty board and the
    low-filled upper bounds have far too many solutions for that; solve a
    book with solving.solve_puzzlebook if its solutions are wanted.

    Always computes -- the grouping (minutes at most), then one timed solve
    per bound puzzle, dominated by the upper book's low-filled puzzles.
    Nothing is read or written: keep the result with
    serialization.save_bounds, read it back with serialization.load_bounds."""
    empty = empty_solution.puzzle
    if empty.nr_filled_cells:
        raise ValueError(f"Puzzle {empty.name} isn't empty: it has {empty.nr_filled_cells} filled cells.")
    if empty_stats.puzzle_name != empty_solution.puzzle_name:
        raise ValueError(f"The stats are of {empty_stats.puzzle_name}, the solutions of {empty_solution.puzzle_name}.")
    if empty_stats.complete is not True:
        raise ValueError(f"The solve of {empty.name} isn't known to be complete, so it may miss solutions.")
    if not len(empty_solution):
        raise ValueError("The empty board has no solutions, so no puzzle on this board does.")

    setup = empty_solution.setup
    block_idxs = list(setup.blocks)
    counts = bound_counts(
        empty_solution.rows,
        [len(setup.placement_cells[idx]) for idx in block_idxs],
        [setup.blocks[idx].count for idx in block_idxs],
        verbose=verbose,
    )
    bounds = []
    for bound, name, difficulty in zip(
        counts, (LOWER_BOUND_BOOK, UPPER_BOUND_BOOK), (HARDEST_DIFFICULTY, EASIEST_DIFFICULTY)
    ):
        puzzles = _bound_puzzles(setup, bound, name, difficulty)
        expected = [bound[filled][0] for filled in sorted(bound)]
        bounds.append((puzzles, _time_bound_book(puzzles, empty_stats, verbose, expected)))
    return bounds[0], bounds[1]
