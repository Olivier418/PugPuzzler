"""Solving puzzles and packaging the outcome: run the solver over a Puzzle /
PuzzleBook, time it, wrap the results in Solution/SolveStats containers and
(opt-in, `save_solutions`/`save_stats`) save them to a folder mirroring where
the puzzle was loaded from. Every solve here is timed by timed_solve, on the
solver's fast rows path. Sits above both `classes` (the
data model) and `serialization` (disk I/O), so neither has to import the
other."""
import math
import time
from enum import IntEnum
from pathlib import Path

import numpy as np
from tqdm import tqdm

from classes import Game, Puzzle, PuzzleBook, Setup, Solution, SolutionBook, SolveStats, SolveStatsBook
from constants import SOLUTION_DIR, UNPLACED
from serialization import save_run
from serialization.paths import next_run_dir


class Verbosity(IntEnum):
    """How much solve_puzzle/solve_puzzlebook print (plain ints work too).
    Every level >= SUMMARY ends with the summary line; the levels above it
    differ in what is printed per solution, one style each (never both, as the
    rendered puzzle's header already names the puzzle and solution number)."""
    SILENT = 0          # nothing
    SUMMARY = 1         # only "Found 21 solutions to puzzle x in 1.23s", once at the end
                        # (at SHOW_SOLUTIONS the puzzle name is dropped if any were rendered)
    EACH_SOLUTION = 2   # plus "Found 3rd solution to puzzle x in 0.42s" as each one is found
    SHOW_SOLUTIONS = 3  # plus the solved puzzle itself, instead of the line above


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def single_run_books(solution: Solution, stats: SolveStats) -> tuple[SolutionBook, SolveStatsBook]:
    """Wrap one puzzle's (Solution, SolveStats) in book containers, which is
    what buys save_run/load_run compatibility."""
    source = solution.puzzle.source
    book_name = source.book_name if source else None
    return (
        SolutionBook(solution, book_name=book_name),
        SolveStatsBook(stats, book_name=book_name, options=stats.options, seed=stats.seed),
    )


class _RowBuffer:
    """Solution rows gathered one at a time into a doubling numpy buffer:
    an empty board can have millions of solutions, and a list of that many
    small arrays would cost ~10x the memory."""

    def __init__(self, setup: Setup):
        self._buf = np.empty((1024, len(setup.blocks)), dtype=setup.row_dtype)
        self._n = 0

    def append(self, row: np.ndarray):
        if self._n == len(self._buf):
            self._buf = np.concatenate([self._buf, np.empty_like(self._buf)])
        self._buf[self._n] = row
        self._n += 1

    def array(self) -> np.ndarray:
        return self._buf[:self._n].copy()


def _state_from_row(puzzle: Puzzle, row: np.ndarray, nr: int) -> Puzzle:
    """A solution row as the fully placed copy of `puzzle` that
    Puzzle.solve would have yielded, named "<name> (solution <nr>)"."""
    state = puzzle.copy(rename=f"{puzzle.name} (solution {nr})" if puzzle.name else None)
    for idx, placement_idx in zip(puzzle.blocks, row):
        if state.chosen_placement_idx[idx] == UNPLACED:
            state.place_unchecked(idx, int(placement_idx))
    return state


def timed_solve(
    puzzle: Puzzle,
    seed: int = None,
    verbose: int = Verbosity.SILENT,
    time_limit: float = math.inf,
    max_solutions: float = math.inf,
    keep_rows: bool = True,
    progress: bool = False,
    **options,
) -> tuple[np.ndarray | None, SolveStats]:
    """Solve `puzzle` on the clock every solve is timed with, and return
    its solutions as rows (see Solver.solve_rows; None unless `keep_rows`)
    plus the run's SolveStats, `complete` included. The one timing path
    shared by solve_puzzle and puzzle_bounds, so their times compare.

    What is on the clock: the search, one elapsed stamp per solution, and
    whatever `verbose` prints per solution (at SHOW_SOLUTIONS a State is
    built from the row to print it). What isn't: numba's warmup and
    anything done with the rows afterwards (e.g. building grids).
    `progress` shows a tqdm bar of solutions found."""
    # Kept out of the clock: a one-off per-process cost, not search time.
    puzzle.setup.warmup()
    rows = _RowBuffer(puzzle.setup) if keep_rows else None
    elapsed = []
    solutions = puzzle.solve_rows(seed=seed, time_limit=time_limit, max_solutions=max_solutions, **options)
    bar = tqdm(desc=f"Solving puzzle {puzzle.name}", unit=" solutions", disable=not progress)
    start = time.perf_counter()
    while True:
        try:
            row = next(solutions)
        except StopIteration as done:  # a for loop would drop its value
            complete = done.value
            break
        elapsed.append(time.perf_counter() - start)
        bar.update()
        if rows is not None:
            rows.append(row)
        if verbose >= Verbosity.SHOW_SOLUTIONS:
            print(_state_from_row(puzzle, row, len(elapsed)))
        elif verbose >= Verbosity.EACH_SOLUTION:
            print(f"Found {_ordinal(len(elapsed))} solution to puzzle {puzzle.name} in {elapsed[-1]:.2f}s")
    duration = time.perf_counter() - start
    bar.close()

    if verbose >= Verbosity.SUMMARY:
        noun = "solution" if len(elapsed) == 1 else "solutions"
        # Rendered puzzles carry their own name in the header; only repeat it
        # when nothing was rendered above.
        target = "" if elapsed and verbose >= Verbosity.SHOW_SOLUTIONS else f" to puzzle {puzzle.name}"
        print(f"Found {len(elapsed)} {noun}{target} in {duration:.2f}s")

    stats = SolveStats(
        puzzle_name=puzzle.name, options=dict(options), seed=seed, duration=duration, elapsed=elapsed,
        complete=complete,
    )
    return (rows.array() if rows is not None else None), stats


def empty_puzzle(game: Game, board_key: str) -> Puzzle:
    """`game`'s empty puzzle on its `board_key` board, the standalone
    games/<game>/puzzles/empty_<board_key>.json every game ships."""
    name = f"empty_{board_key}"
    puzzle = game.puzzles.get(name)
    if puzzle is None or puzzle.setup is not game.setups[board_key] or puzzle.nr_filled_cells:
        raise ValueError(
            f"Game {game.name!r} has no empty puzzle {name!r} on board {board_key!r} "
            f"(games/{game.name}/puzzles/{name}.json)."
        )
    return puzzle


def solve_puzzle(
    puzzle: Puzzle,
    seed: int = None,
    verbose: int = Verbosity.SILENT,
    save_solutions: bool = False,
    save_stats: bool = False,
    path: str | Path = None,
    solutions_root: str | Path = SOLUTION_DIR,
    time_limit: float = math.inf,
    max_solutions: float = math.inf,
    progress: bool = False,
    **options,
) -> tuple[Solution, SolveStats]:
    """Solve a single Puzzle and save the solved grids (`save_solutions`)
    and/or the run's SolveStats (`save_stats`) to a folder mirroring where
    the Puzzle itself was loaded from, under `solutions_root` (e.g.
    games/IQpuzzler/puzzles/main_empty ->
    solutions/IQpuzzler/puzzles/main_empty/result_<idx>/{solutions,stats}.json).

    Saving is off by default; the two flags are independent, since the
    grids are the bulk of a save and the stats are useful without them.
    Pass `path=` to save somewhere specific instead of the mirrored default
    (`path` alone does not save). `verbose` (see Verbosity)
    controls progress printing. `time_limit` (seconds) and
    `max_solutions` stop the solve early, whichever is hit first (both
    default to infinity; not recorded in the SolveStats -- its
    `duration` and `elapsed` show a truncated run). They are named here
    rather than left in `options` because they change which solutions
    come back, not how they are found. `options` are forwarded to the
    solver and recorded in the SolveStats.

    Timed by timed_solve (`progress` shows its bar); the stats say whether
    the run is `complete`.
    """
    rows, stats = timed_solve(
        puzzle, seed=seed, verbose=verbose, time_limit=time_limit, max_solutions=max_solutions,
        progress=progress, **options,
    )
    solution = Solution(puzzle, rows)

    if save_solutions or save_stats:
        target = Path(path) if path is not None else next_run_dir(puzzle.source, solutions_root)
        solution_book, stats_book = single_run_books(solution, stats)
        save_run(
            solution_book if save_solutions else None, stats_book if save_stats else None, target,
        )

    return solution, stats


def solve_puzzlebook(
    puzzlebook: PuzzleBook,
    seed: int = None,
    verbose: int = Verbosity.SILENT,
    save_solutions: bool = False,
    save_stats: bool = False,
    path: str | Path = None,
    solutions_root: str | Path = SOLUTION_DIR,
    time_limit: float = math.inf,
    max_solutions: float = math.inf,
    **options,
) -> tuple[SolutionBook, SolveStatsBook]:
    """Solve every puzzle in a PuzzleBook and save the combined solutions
    (`save_solutions`) and/or solve stats (`save_stats`) to a folder
    mirroring where the book itself was loaded from (see solve_puzzle for
    the mirroring rule). `verbose` applies to each puzzle in turn.

    `time_limit` and `max_solutions` apply to each puzzle separately.

    Delegates per-puzzle solving to solve_puzzle so the two entry points
    can't drift apart; only the batching and the single combined save are
    specific to this function.
    """
    solutions = []
    stats_list = []

    for puzzle in puzzlebook.values():
        solution, stats = solve_puzzle(
            puzzle,
            seed=seed,
            verbose=verbose,
            time_limit=time_limit,
            max_solutions=max_solutions,
            **options,
        )
        solutions.append(solution)
        stats_list.append(stats)

    solution_book = SolutionBook(*solutions, book_name=puzzlebook.name)
    stats_book = SolveStatsBook(*stats_list, book_name=puzzlebook.name, options=options, seed=seed)

    if save_solutions or save_stats:
        target = Path(path) if path is not None else next_run_dir(puzzlebook.source, solutions_root)
        save_run(
            solution_book if save_solutions else None, stats_book if save_stats else None, target, flat=False,
        )

    return solution_book, stats_book
