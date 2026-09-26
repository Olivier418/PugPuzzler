"""A run folder: the outcome of one solve, saved as solutions.json and/or
stats.json (either half can be left out, see solving.solve_puzzle's
`save_solutions`/`save_stats`). It stores no puzzles -- its path, which
mirrors the puzzles' Source (see Source.relative_dir), points at them in
games/, exactly as it relies on the game's blocks and boards there. Loaded
back, a run is three parallel books keyed by puzzle name: the puzzles
(always), the solutions and the stats (each if saved)."""
from functools import lru_cache
from pathlib import Path

from classes import Puzzle, PuzzleBook, Setup, Solution, SolutionBook, SolveStats, SolveStatsBook
from constants import GAMES_DIR, SOLUTION_DIR
from .loading import load_book, load_setups, load_standalone_puzzles
from .paths import game_book_from_run_dir, puzzle_name_from_run_dir
from .solutions_io import load_solutionbook, write_solutions
from .stats_io import load_solve_stats_book, write_solve_stats

SOLUTIONS_FILE = "solutions.json"
STATS_FILE = "stats.json"


def save_run(
    solution_book: SolutionBook | None,
    stats_book: SolveStatsBook | None,
    folder: str | Path,
    flat: bool = True,
) -> Path:
    """Save a SolutionBook and/or the SolveStatsBook of the same solve as one
    run folder; a None half is simply not written. `flat=True` (the
    default) is for one puzzle's run, saved below that puzzle's own folder
    (`.../<puzzle_name>/<idx>`), which is where its name is recovered from
    on load; a whole book's run needs `flat=False` to keep the puzzle names
    in the files."""
    if solution_book is None and stats_book is None:
        raise ValueError("Nothing to save: both the SolutionBook and the SolveStatsBook are None.")
    if solution_book is not None and stats_book is not None and solution_book.keys() != stats_book.keys():
        raise ValueError("The SolutionBook and SolveStatsBook must cover the same puzzles.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if solution_book is not None:
        write_solutions(solution_book, folder / SOLUTIONS_FILE, flat=flat)
    if stats_book is not None:
        write_solve_stats(stats_book, folder / STATS_FILE, flat=flat)
    return folder


@lru_cache(maxsize=None)
def _game_setups(game_dir: Path) -> dict[str, Setup]:
    """One set of Setups per game per process, shared by every run loaded
    from it."""
    return load_setups(game_dir)


def _run_puzzles(game_dir: Path, book_name: str | None) -> dict[str, Puzzle]:
    """The puzzles a run can refer to: its book's, or the game's standalone ones."""
    setups = _game_setups(game_dir)
    if book_name is not None:
        return load_book(game_dir, book_name, setups)
    return {p.name: p for p in load_standalone_puzzles(game_dir, setups)}


def load_run(
    folder: str | Path, games_root: str | Path = GAMES_DIR
) -> tuple[PuzzleBook, SolutionBook | None, SolveStatsBook | None]:
    """Load a run folder saved by save_run (or solving.solve_puzzle /
    solve_puzzlebook with a save flag) back into (puzzles, solutions,
    stats): a PuzzleBook of the puzzles the run solved, and the SolutionBook
    and SolveStatsBook of that solve, all keyed by the same puzzle names. A
    half that wasn't saved comes back as None; the puzzles never do.

    The folder's path names the game, book and (for a single-puzzle run)
    puzzle -- any root works, as long as it ends in
    `<game>/(books/<book>[/<puzzle>]|puzzles/<puzzle>)/<run>`. The puzzles
    are loaded from that game under `games_root`, whose files never change
    (only new books are added, by save_puzzlebook), so a run can't go stale."""
    folder = Path(folder)
    game_name, book_name = game_book_from_run_dir(folder)
    flat_name = puzzle_name_from_run_dir(folder)
    solutions_file, stats_file = folder / SOLUTIONS_FILE, folder / STATS_FILE
    if not solutions_file.exists() and not stats_file.exists():
        raise FileNotFoundError(f"{folder} holds neither {SOLUTIONS_FILE} nor {STATS_FILE}.")

    lookup = _run_puzzles(Path(games_root) / game_name, book_name)
    solution_book = (
        load_solutionbook(solutions_file, lookup, flat_name, book_name) if solutions_file.exists() else None
    )
    stats_book = load_solve_stats_book(stats_file, flat_name, book_name) if stats_file.exists() else None

    names = list((stats_book if stats_book is not None else solution_book).keys())
    if solution_book is not None and list(solution_book.keys()) != names:
        raise ValueError(f"{solutions_file} and {stats_file} cover different puzzles.")
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"{folder} has results for puzzles {missing}, which {games_root} doesn't have.")
    puzzles = PuzzleBook(*(lookup[name] for name in names), name=book_name or flat_name)
    return puzzles, solution_book, stats_book


def load_puzzle_run(folder: str | Path, puzzle: Puzzle) -> tuple[Solution | None, SolveStats | None]:
    """A single-puzzle run folder of `puzzle` (which must be the puzzle its
    path names), loaded straight onto it rather than onto a fresh copy
    from games/: (Solution, SolveStats), either None if not saved."""
    folder = Path(folder)
    if puzzle_name_from_run_dir(folder) != puzzle.name:
        raise ValueError(f"{folder} is a run of {puzzle_name_from_run_dir(folder)!r}, not of {puzzle.name!r}.")
    solutions_file, stats_file = folder / SOLUTIONS_FILE, folder / STATS_FILE
    solution = stats = None
    if solutions_file.exists():
        solution = load_solutionbook(solutions_file, {puzzle.name: puzzle}, puzzle.name)[puzzle.name]
    if stats_file.exists():
        stats = load_solve_stats_book(stats_file, puzzle.name)[puzzle.name]
    return solution, stats


def latest_complete_run(puzzle: Puzzle, root: str | Path = SOLUTION_DIR) -> Path | None:
    """The newest run folder of `puzzle` under `root` (see
    paths.next_run_dir) that holds its solutions and whose stats say they
    are complete -- every solution, not a time- or count-limited subset;
    None if there is none (or `puzzle` isn't in games/)."""
    if puzzle.source is None:
        return None
    base = Path(root) / puzzle.source.relative_dir()
    runs = [p for p in base.glob("result_*") if p.name.removeprefix("result_").isdigit()] if base.exists() else []
    for folder in sorted(runs, key=lambda p: int(p.name.removeprefix("result_")), reverse=True):
        stats_file = folder / STATS_FILE
        if (folder / SOLUTIONS_FILE).exists() and stats_file.exists():
            if load_solve_stats_book(stats_file, puzzle.name)[puzzle.name].complete:
                return folder
    return None
