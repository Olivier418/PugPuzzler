"""Solving puzzles and packaging the outcome: run the solver over a Puzzle /
PuzzleBook, time it, wrap the results in Solution/SolveStats containers and
(opt-in, `save=True`) save them to a folder mirroring where the puzzle was
loaded from. Sits above both `classes` (the data model) and `serialization` (disk
I/O), so neither has to import the other."""
import math
import time
from enum import IntEnum
from pathlib import Path

import numpy as np

from classes import Puzzle, PuzzleBook, Solution, SolutionBook, SolveStats, SolveStatsBook
from constants import SOLUTION_DIR
from serialization import save_solution_run
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


def make_result(
    puzzle: Puzzle,
    grids: list[np.ndarray],
    elapsed: list[float],
    duration: float,
    seed: int | None,
    options: dict,
) -> tuple[Solution, SolveStats]:
    """Package one solver run of `puzzle` as a (Solution, SolveStats).
    `grids` are full board-shaped (see Setup.to_full_grid), matching the
    on-disk format even though State.grid itself is compact."""
    source = puzzle.source
    solution = Solution(
        puzzle_name=puzzle.name,
        grids=grids,
        game_name=source.game_name if source else None,
        book_name=source.book_name if source else None,
        puzzle=puzzle,
    )
    stats = SolveStats(
        puzzle_name=puzzle.name,
        options=dict(options),
        seed=seed,
        duration=duration,
        elapsed=elapsed,
    )
    return solution, stats


def single_run_books(solution: Solution, stats: SolveStats) -> tuple[SolutionBook, SolveStatsBook]:
    """Wrap one puzzle's (Solution, SolveStats) in book containers, which is
    what buys save_solution_run/load_solution_run compatibility."""
    return (
        SolutionBook(solution, game_name=solution.game_name, book_name=solution.book_name),
        SolveStatsBook(
            stats,
            game_name=solution.game_name,
            book_name=solution.book_name,
            options=stats.options,
            seed=stats.seed,
        ),
    )


def solve_puzzle(
    puzzle: Puzzle,
    seed: int = None,
    verbose: int = Verbosity.SILENT,
    save: bool = False,
    path: str | Path = None,
    solutions_root: str | Path = SOLUTION_DIR,
    time_limit: float = math.inf,
    max_solutions: float = math.inf,
    **options,
) -> tuple[Solution, SolveStats]:
    """Solve a single Puzzle and, with `save=True`, save the results and
    the run's SolveStats to a folder mirroring where the Puzzle
    itself was loaded from, under `solutions_root` (e.g.
    games/IQpuzzler/puzzles/main_empty ->
    solutions/IQpuzzler/puzzles/main_empty/result_<idx>/{solutions,stats}.json).

    Saving is off by default; pass `save=True` to save, and `path=` to
    save somewhere specific instead of the mirrored default (`path` alone
    does not save). `verbose` (see Verbosity)
    controls progress printing. `time_limit` (seconds) and
    `max_solutions` stop the solve early, whichever is hit first (both
    default to infinity; not recorded in the SolveStats -- its
    `duration` and `elapsed` show a truncated run). They are named here
    rather than left in `options` because they change which solutions
    come back, not how they are found. `options` are forwarded to the
    solver and recorded in the SolveStats.
    """
    # Kept out of the clock: a one-off per-process cost, not search time.
    puzzle.setup.warmup()
    start = time.perf_counter()
    grids = []
    elapsed = []
    for state in puzzle.solve(
        seed=seed,
        time_limit=time_limit,
        max_solutions=max_solutions,
        **options,
    ):
        grids.append(state.setup.to_full_grid(state.grid))
        elapsed.append(time.perf_counter() - start)
        if verbose >= Verbosity.SHOW_SOLUTIONS:
            print(state)
        elif verbose >= Verbosity.EACH_SOLUTION:
            print(f"Found {_ordinal(len(grids))} solution to puzzle {puzzle.name} in {elapsed[-1]:.2f}s")
    duration = time.perf_counter() - start

    if verbose >= Verbosity.SUMMARY:
        noun = "solution" if len(grids) == 1 else "solutions"
        # Rendered puzzles carry their own name in the header; only repeat it
        # when nothing was rendered above.
        target = "" if grids and verbose >= Verbosity.SHOW_SOLUTIONS else f" to puzzle {puzzle.name}"
        print(f"Found {len(grids)} {noun}{target} in {duration:.2f}s")

    solution, stats = make_result(puzzle, grids, elapsed, duration, seed, options)

    if save:
        target = Path(path) if path is not None else next_run_dir(puzzle.source, solutions_root)
        save_solution_run(*single_run_books(solution, stats), target)

    return solution, stats


def solve_puzzlebook(
    puzzlebook: PuzzleBook,
    game_name: str = None,
    seed: int = None,
    verbose: int = Verbosity.SILENT,
    save: bool = False,
    path: str | Path = None,
    solutions_root: str | Path = SOLUTION_DIR,
    time_limit: float = math.inf,
    max_solutions: float = math.inf,
    **options,
) -> tuple[SolutionBook, SolveStatsBook]:
    """Solve every puzzle in a PuzzleBook and, with `save=True`, save the
    combined solutions and solve stats to a folder mirroring where
    the book itself was loaded from (see solve_puzzle for the
    mirroring rule). `verbose` applies to each puzzle in turn.

    `time_limit` and `max_solutions` apply to each puzzle separately.

    Delegates per-puzzle solving to solve_puzzle so the two entry points
    can't drift apart; only the batching and the single combined save are
    specific to this function.
    """
    source = puzzlebook.source
    book_name = puzzlebook.name
    game_name = game_name or (source.game_name if source else None)
    solutions = []
    stats_list = []

    for puzzle in puzzlebook.values():
        solution, stats = solve_puzzle(
            puzzle,
            seed=seed,
            verbose=verbose,
            save=False,
            time_limit=time_limit,
            max_solutions=max_solutions,
            **options,
        )
        # Individual puzzles carry their own Source, but this
        # SolutionBook is filed under the book's own game/book name --
        # keep every Solution in it consistent with that.
        solution.game_name = game_name
        solution.book_name = book_name
        solutions.append(solution)
        stats_list.append(stats)

    solution_book = SolutionBook(*solutions, game_name=game_name, book_name=book_name)
    stats_book = SolveStatsBook(
        *stats_list, game_name=game_name, book_name=book_name, options=options, seed=seed
    )

    if save:
        target = Path(path) if path is not None else next_run_dir(source, solutions_root)
        save_solution_run(solution_book, stats_book, target, flat=False)

    return solution_book, stats_book
