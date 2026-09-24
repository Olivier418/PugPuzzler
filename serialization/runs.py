"""A run folder: the outcome of one solve (or one benchmark trial), saved as
solutions.json + stats.json. It stores no puzzles -- its path, which
mirrors the puzzles' Source (see Source.relative_dir), points at them in
games/, exactly as it relies on the game's blocks and boards there."""
from functools import lru_cache
from pathlib import Path

from classes import Game, Puzzle, Setup, SolutionBook, SolveStatsBook
from constants import GAMES_DIR
from .loading import load_book, load_setups, load_standalone_puzzles
from .paths import game_book_from_run_dir, puzzle_name_from_run_dir
from .solutions_io import load_solutionbook, write_solutions
from .stats_io import load_solve_stats_book, write_solve_stats

SOLUTIONS_FILE = "solutions.json"
STATS_FILE = "stats.json"


def save_solution_run(
    solution_book: SolutionBook, stats_book: SolveStatsBook, folder: str | Path, flat: bool = True
) -> Path:
    """Save a SolutionBook and the SolveStatsBook of the same solve as one
    run folder. `flat=True` (the default) is for one puzzle's run, saved
    below that puzzle's own folder (`.../<puzzle_name>/<idx>`), which is
    where its name is recovered from on load; any other save -- a whole
    book, or benchmark.py's trial folders -- needs `flat=False` to keep
    the puzzle names in the files."""
    if solution_book.keys() != stats_book.keys():
        raise ValueError("The SolutionBook and SolveStatsBook must cover the same puzzles.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    write_solutions(solution_book, folder / SOLUTIONS_FILE, flat=flat)
    write_solve_stats(stats_book, folder / STATS_FILE, flat=flat)
    return folder


@lru_cache(maxsize=None)
def _game_setups(game_dir: Path) -> dict[str, Setup]:
    """One set of Setups per game per process, shared by every run loaded
    from it (a benchmark is dozens of runs of one puzzle)."""
    return load_setups(game_dir)


def _run_puzzles(game_name: str, book_name: str | None, game: Game | None) -> dict[str, Puzzle]:
    """The puzzles a run can refer to: its book's, or the game's standalone ones."""
    if game is not None:
        if book_name is None:
            return game.puzzles
        if book_name not in game.books:
            raise ValueError(f"Game {game.name!r} has no book {book_name!r}.")
        return game.books[book_name]
    game_dir = GAMES_DIR / game_name
    setups = _game_setups(game_dir)
    if book_name is not None:
        return load_book(game_dir, book_name, setups)
    return {p.name: p for p in load_standalone_puzzles(game_dir, setups)}


def load_solution_run(folder: str | Path, game: Game = None) -> tuple[SolutionBook, SolveStatsBook]:
    """Load a run folder saved by save_solution_run (or
    solving.solve_puzzle/solve_puzzlebook with save=True) back into its
    (SolutionBook, SolveStatsBook) pair, every Solution holding its live
    Puzzle.

    The folder's path names the game, book and (for a single-puzzle run)
    puzzle -- any root works, as long as it ends in
    `<game>/(books/<book>[/<puzzle>]|puzzles/<puzzle>)/<run>`. The puzzles
    come from `game` when given (sharing the Puzzle objects already in
    memory), else from the game's files in games/, which never change
    (only new books are added, by save_puzzlebook)."""
    folder = Path(folder)
    game_name, book_name = game_book_from_run_dir(folder)
    puzzles = _run_puzzles(game_name, book_name, game)
    flat_name = puzzle_name_from_run_dir(folder)
    return (
        load_solutionbook(folder / SOLUTIONS_FILE, puzzles, flat_name, book_name),
        load_solve_stats_book(folder / STATS_FILE, flat_name, book_name),
    )
