from pathlib import Path

from classes import SolutionBook, SolveStatsBook
from .solutions_io import load_solutionbook, write_solutions
from .stats_io import load_solve_stats_book, write_solve_stats


def save_solution_run(
    solution_book: SolutionBook, stats_book: SolveStatsBook, folder: str | Path, flat: bool = True
) -> Path:
    """Save a Solution/SolveStats pair produced by the same solve into
    one run folder, as solutions.json + stats.json side by side.

    `flat=True` (the default) drops the "puzzles" wrapper for a
    single-puzzle result (solving.solve_puzzle, standalone or out of a
    book) -- puzzle_name is recovered on load from `folder` itself
    (`.../<puzzle_name>/<idx>`), same as game_name/book_name are from the
    rest of the path (see Source.relative_dir()). Callers saving more
    than one puzzle's results into one file -- a whole book solved via
    solving.solve_puzzlebook, or benchmark.py's per-trial folders that
    don't follow the `<puzzle_name>/<idx>` convention -- must pass
    `flat=False` to keep puzzle_name explicit in the JSON.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_solutions(solution_book, folder / "solutions.json", flat=flat)
    write_solve_stats(stats_book, folder / "stats.json", flat=flat)
    return folder


def load_solution_run(folder: str | Path) -> tuple[SolutionBook, SolveStatsBook]:
    """Load a run folder saved by save_solution_run (or
    solving.solve_puzzle/solve_puzzlebook with save=True) back into its
    (SolutionBook, SolveStatsBook) pair."""
    folder = Path(folder)
    return load_solutionbook(folder / "solutions.json"), load_solve_stats_book(folder / "stats.json")
