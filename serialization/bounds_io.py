"""A board's solution bounds on disk, `bounds/<game>/<board_key>/`: the two
books puzzle_bounds.compute_puzzle_bounds derives from the empty board's
solutions, saved with save_bounds and read back with load_bounds.

  lower_puzzles.json   the lower bound's puzzles, in the book-file format
  lower_stats.json     their SolveStats (flat=False), the empty board's included
  upper_puzzles.json   the same for the upper bound
  upper_stats.json

Letter grids describe themselves and games/ never changes, so -- like a run
in solutions/ -- a bounds folder can't go stale."""
from pathlib import Path

from classes import Game, PuzzleBook, SolveStatsBook
from constants import BOUNDS_DIR, LOWER_BOUND_BOOK, UPPER_BOUND_BOOK
from .puzzles_io import load_puzzles, write_puzzles
from .stats_io import load_solve_stats_book, write_solve_stats

Bounds = tuple[tuple[PuzzleBook, SolveStatsBook], tuple[PuzzleBook, SolveStatsBook]]

_SIDES = (("lower", LOWER_BOUND_BOOK), ("upper", UPPER_BOUND_BOOK))


def _book_files(folder: Path, side: str) -> tuple[Path, Path]:
    return folder / f"{side}_puzzles.json", folder / f"{side}_stats.json"


def bounds_dir(game_name: str, board_key: str, root: str | Path = BOUNDS_DIR) -> Path:
    """The bounds folder of `game_name`'s `board_key` board."""
    return Path(root) / game_name / board_key


def save_bounds(
    game: Game,
    lower: tuple[PuzzleBook, SolveStatsBook],
    upper: tuple[PuzzleBook, SolveStatsBook],
    root: str | Path = BOUNDS_DIR,
) -> Path:
    """Write both bounds' (PuzzleBook, SolveStatsBook) to the bounds folder
    of the board they are on (one of `game`'s Setups), overwriting what is
    there. Returns the folder."""
    board_key = game.board_key(lower[0].setup)
    if upper[0].setup is not lower[0].setup:
        raise ValueError("The lower and upper bound books are on different boards.")
    folder = bounds_dir(game.name, board_key, root)
    folder.mkdir(parents=True, exist_ok=True)
    for (side, _), (puzzles, stats) in zip(_SIDES, (lower, upper)):
        puzzle_file, stats_file = _book_files(folder, side)
        write_puzzles(puzzles.values(), board_key, puzzle_file)
        write_solve_stats(stats, stats_file, flat=False)
    return folder


def load_bounds(game: Game, board_key: str, root: str | Path = BOUNDS_DIR) -> Bounds:
    """Both bounds' (PuzzleBook, SolveStatsBook) as save_bounds wrote them
    for `game`'s `board_key` board. Raises FileNotFoundError unless all four
    files are there."""
    folder = bounds_dir(game.name, board_key, root)
    missing = [str(f) for side, _ in _SIDES for f in _book_files(folder, side) if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"No saved bounds for {game.name} {board_key}: missing {', '.join(missing)}. "
            f"Compute them with puzzle_bounds.compute_puzzle_bounds and save them with save_bounds."
        )
    bounds = []
    for side, name in _SIDES:
        puzzle_file, stats_file = _book_files(folder, side)
        puzzles = PuzzleBook(*load_puzzles(puzzle_file, {board_key: game.setups[board_key]}), name=name)
        bounds.append((puzzles, load_solve_stats_book(stats_file, None, name)))
    return bounds[0], bounds[1]
