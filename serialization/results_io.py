"""What solving produces, on disk: runs in solutions/, benchmark trials in
benchmarks/ (written by benchmark.py) and a board's bounds in bounds/.

Two file formats, each one layout whether it holds one puzzle or a book:

  solutions.json  [{"puzzle_name", "grids": [letter grid, ...]}, ...]
  stats.json      [{"puzzle_name", "options", "seed", "duration", "complete", "elapsed"}, ...]

and a book file (games_io.write_puzzles) for the bounds' puzzles. Where
they live:

  run      <root>/<Source.relative_dir()>/result_<idx>/{solutions.json, stats.json}
           (either file may be left out; see solving.solve_puzzle)
  benchmark <root>/<Source.relative_dir()>/benchmark_<idx>/<config label>/stats<seed>.json
  bounds   <root>/<game>/<board_key>/{lower,upper}_{puzzles,stats}.json

A run stores no puzzles, and no game or book name: its path names the game
and book (mirroring the puzzles' Source), the files name the puzzles, and
the puzzles are loaded from games/, which never changes (save_puzzlebook
only adds books) -- so a run can't go stale. Neither can a bounds folder,
whose letter grids describe themselves."""
import json
from collections.abc import Mapping
from pathlib import Path

from classes import Game, Puzzle, PuzzleBook, Solution, SolveStats, Source
from constants import BOUNDS_DIR, GAMES_DIR, LOWER_BOUND_BOOK, UNPLACED, UPPER_BOUND_BOOK
from .games_io import load_book, load_puzzles, load_setups, load_standalone_puzzles, write_puzzles
from .jsonio import block_grids_to_rows, dump_json, rows_to_block_grids

SOLUTIONS_FILE = "solutions.json"
STATS_FILE = "stats.json"


# ---- paths ------------------------------------------------------------------

def next_run_dir(source: Source | None, root: str | Path, prefix: str = "result") -> Path:
    """Where to auto-save a run: `source`'s game/puzzle-or-book directory
    mirrored under `root`, then `<prefix>_<idx>` one past the highest idx
    there -- re-solving the same puzzle/book is expected to reproduce the
    same solutions, but every run still gets its own folder, and the
    newest always has the highest idx."""
    if source is None:
        raise ValueError(
            "Can't save a run of a Puzzle/PuzzleBook that isn't in games/ (no `source`): a run "
            "refers to its puzzles there. Give the book a home first with "
            "serialization.save_puzzlebook."
        )
    base = Path(root) / source.relative_dir()
    taken = [p.name.removeprefix(f"{prefix}_") for p in base.glob(f"{prefix}_*")] if base.exists() else []
    return base / f"{prefix}_{max((int(idx) for idx in taken if idx.isdigit()), default=0) + 1}"


def _game_book(folder: Path) -> tuple[str, str | None]:
    """(game_name, book_name) of a run folder -- the inverse of
    Source.relative_dir(), book_name None for a standalone puzzle: the
    folder's nearest `books`/`puzzles` component sits right below the game."""
    parts = Path(folder).parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == "puzzles" or (parts[i] == "books" and i + 2 < len(parts)):
            return parts[i - 1], parts[i + 1] if parts[i] == "books" else None
    raise ValueError(
        f"Can't tell which puzzles the run in {folder} belongs to: its path doesn't follow "
        "<game>/(books/<book>|puzzles)/..."
    )


# ---- solutions.json and stats.json -----------------------------------------------

# Solutions are turned into grids this many at a time on the way out, which
# bounds the memory a huge Solution (millions of rows) needs to be written.
_CHUNK = 1 << 16


def _letter_grids(sol: Solution) -> list:
    """`sol`'s solutions as letter rows, one entry per solution."""
    letters = []
    for start in range(0, len(sol), _CHUNK):
        grids = sol.setup.rows_to_grids(sol.rows[start:start + _CHUNK])
        letters += block_grids_to_rows(grids, sol.puzzle.blocks)
    return letters


def write_solutions(solutions: Mapping[str, Solution], file: Path) -> None:
    """Every solution as a letter grid (one string per row), so the file
    can be read by eye, under its puzzle's name."""
    dump_json([{"puzzle_name": sol.puzzle_name, "grids": _letter_grids(sol)} for sol in solutions.values()], file)


def load_solutions(file: Path, puzzles: Mapping[str, Puzzle]) -> dict[str, Solution]:
    """Inverse of write_solutions: each entry's grids attached to its Puzzle
    in `puzzles`. Every grid is checked to be a solution (see
    Setup.grids_to_rows)."""
    with open(file, "r") as f:
        data = json.load(f)
    solutions = {}
    for entry in data:
        name = entry["puzzle_name"]
        if name not in puzzles:
            raise ValueError(f"{file} holds solutions of puzzle {name!r}, which isn't among {sorted(puzzles)}.")
        puzzle = puzzles[name]
        grids = rows_to_block_grids(entry["grids"], puzzle.blocks, puzzle.board.cells.shape)
        solutions[name] = Solution(puzzle, puzzle.setup.grids_to_rows(grids))
    return solutions


def write_stats(stats: Mapping[str, SolveStats], file: Path) -> None:
    """Every SolveStats as one object of its fields."""
    dump_json([s._asdict() for s in stats.values()], file)


def load_stats(file: Path) -> dict[str, SolveStats]:
    """Inverse of write_stats, keyed by puzzle name."""
    with open(file, "r") as f:
        return {entry["puzzle_name"]: SolveStats(**entry) for entry in json.load(f)}


# ---- runs -----------------------------------------------------------------------

def save_run(
    solutions: Mapping[str, Solution] | None,
    stats: Mapping[str, SolveStats] | None,
    folder: str | Path,
) -> Path:
    """Save the solutions and/or stats of one solve (one puzzle's or a
    book's, keyed by puzzle name) as a run folder; a None half is simply
    not written."""
    if solutions is None and stats is None:
        raise ValueError("Nothing to save: both the solutions and the stats are None.")
    if solutions is not None and stats is not None and solutions.keys() != stats.keys():
        raise ValueError("The solutions and stats must cover the same puzzles.")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if solutions is not None:
        write_solutions(solutions, folder / SOLUTIONS_FILE)
    if stats is not None:
        write_stats(stats, folder / STATS_FILE)
    return folder


def _run_puzzles(game_dir: Path, book_name: str | None) -> dict[str, Puzzle]:
    """The puzzles a run can refer to: its book's, or the game's standalone
    ones -- on the game's shared Setups (see games_io.load_setups)."""
    setups = load_setups(game_dir)
    if book_name is not None:
        return load_book(game_dir, book_name, setups)
    return {p.name: p for p in load_standalone_puzzles(game_dir, setups)}


def load_run_books(
    folder: str | Path, games_root: str | Path = GAMES_DIR
) -> tuple[PuzzleBook, dict[str, Solution] | None, dict[str, SolveStats] | None]:
    """Load a run folder saved by save_run (or solving.solve_puzzle /
    solve_puzzlebook with a save flag) back into (puzzles, solutions,
    stats): a PuzzleBook of the puzzles the run solved, and that solve's
    solutions and stats, all keyed by the same puzzle names. A half that
    wasn't saved comes back as None; the puzzles never do.

    The folder's path names the game and book -- any root works, as long
    as it ends in `<game>/(books/<book>[/<puzzle>]|puzzles/<puzzle>)/<run>`.
    The puzzles are loaded from that game under `games_root`, on the same
    Setups as a load_game of it (see games_io.load_setups), so they can go
    straight into save_bounds / save_puzzlebook with that Game."""
    folder = Path(folder)
    game_name, book_name = _game_book(folder)
    solutions_file, stats_file = folder / SOLUTIONS_FILE, folder / STATS_FILE
    if not solutions_file.exists() and not stats_file.exists():
        raise FileNotFoundError(f"{folder} holds neither {SOLUTIONS_FILE} nor {STATS_FILE}.")

    lookup = _run_puzzles(Path(games_root) / game_name, book_name)
    solutions = load_solutions(solutions_file, lookup) if solutions_file.exists() else None
    stats = load_stats(stats_file) if stats_file.exists() else None

    names = list((stats if stats is not None else solutions).keys())
    if solutions is not None and list(solutions) != names:
        raise ValueError(f"{solutions_file} and {stats_file} cover different puzzles.")
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"{folder} has results for puzzles {missing}, which {games_root} doesn't have.")
    puzzles = PuzzleBook(*(lookup[name] for name in names), name=book_name or names[0])
    return puzzles, solutions, stats


def load_complete_run(folder: str | Path, puzzle: Puzzle) -> tuple[Solution, SolveStats]:
    """`puzzle`'s (Solution, SolveStats) from `folder`, a run folder of it
    that you pick (as solve_puzzle(puzzle, save_solutions=True,
    save_stats=True) writes one), after checking that it really is a
    complete run of `puzzle`:

      - the folder sits where runs of `puzzle` go (<root>/<its
        Source.relative_dir()>/<run>) and its files hold `puzzle` alone;
      - both files are there, and the stats say the run was complete --
        every solution, not a time- or count-limited subset;
      - every solution places `puzzle`'s pre-filled pieces where it has them.

    Raises otherwise. The Solution is attached to `puzzle` itself (the
    cheap checks run before the solutions are read)."""
    folder = Path(folder)
    if puzzle.source is None:
        raise ValueError(f"Puzzle {puzzle.name!r} isn't in games/ (no `source`), so there is no run of it.")
    own = puzzle.source.relative_dir()
    if folder.parent.parts[-len(own.parts):] != own.parts:
        raise ValueError(f"{folder} isn't a run of puzzle {puzzle.name!r}: those live in <root>/{own.as_posix()}/<run>.")
    solutions_file, stats_file = folder / SOLUTIONS_FILE, folder / STATS_FILE
    missing = [f.name for f in (solutions_file, stats_file) if not f.exists()]
    if missing:
        raise FileNotFoundError(f"{folder} lacks {' and '.join(missing)}; a complete run needs both.")

    stats = load_stats(stats_file)
    if list(stats) != [puzzle.name]:
        raise ValueError(f"{folder} is a run of {list(stats)}, not of puzzle {puzzle.name!r} alone.")
    if not stats[puzzle.name].complete:
        raise ValueError(f"{folder} was cut short by a time or solution limit, so it may miss solutions.")

    solution = load_solutions(solutions_file, {puzzle.name: puzzle})[puzzle.name]
    for col, placement_idx in enumerate(puzzle.chosen_placement_idx.values()):
        if placement_idx != UNPLACED and (solution.rows[:, col] != placement_idx).any():
            raise ValueError(f"{folder}'s solutions don't keep puzzle {puzzle.name!r}'s pre-filled pieces in place.")
    return solution, stats[puzzle.name]


# ---- bounds ----------------------------------------------------------------------

Bounds = tuple[tuple[PuzzleBook, dict[str, SolveStats]], tuple[PuzzleBook, dict[str, SolveStats]]]

_SIDES = (("lower", LOWER_BOUND_BOOK), ("upper", UPPER_BOUND_BOOK))


def _bound_files(game_name: str, board_key: str, side: str, root: str | Path) -> tuple[Path, Path]:
    """(puzzles file, stats file) of one side of a board's bounds."""
    folder = Path(root) / game_name / board_key
    return folder / f"{side}_puzzles.json", folder / f"{side}_stats.json"


def save_bounds(
    game: Game,
    lower: tuple[PuzzleBook, dict[str, SolveStats]],
    upper: tuple[PuzzleBook, dict[str, SolveStats]],
    root: str | Path = BOUNDS_DIR,
) -> Path:
    """Write both bounds' (PuzzleBook, stats), as
    puzzle_bounds.compute_puzzle_bounds returns them, to the bounds folder
    of the board they are on (one of `game`'s Setups), overwriting what is
    there. Returns the folder."""
    board_key = game.board_key(lower[0].setup)
    if upper[0].setup is not lower[0].setup:
        raise ValueError("The lower and upper bound books are on different boards.")
    for (side, _), (puzzles, stats) in zip(_SIDES, (lower, upper)):
        puzzle_file, stats_file = _bound_files(game.name, board_key, side, root)
        puzzle_file.parent.mkdir(parents=True, exist_ok=True)
        write_puzzles(puzzles.values(), board_key, puzzle_file)
        write_stats(stats, stats_file)
    return puzzle_file.parent


def load_bounds(game: Game, board_key: str, root: str | Path = BOUNDS_DIR) -> Bounds:
    """Both bounds' (PuzzleBook, stats) as save_bounds wrote them for
    `game`'s `board_key` board, the books named LOWER_BOUND_BOOK /
    UPPER_BOUND_BOOK. Raises FileNotFoundError unless all four files are
    there."""
    files = [_bound_files(game.name, board_key, side, root) for side, _ in _SIDES]
    missing = [str(f) for pair in files for f in pair if not f.exists()]
    if missing:
        raise FileNotFoundError(
            f"No saved bounds for {game.name} {board_key}: missing {', '.join(missing)}. "
            f"Compute them with puzzle_bounds.compute_puzzle_bounds and save them with save_bounds."
        )
    lower, upper = (
        (PuzzleBook(*load_puzzles(puzzle_file, {board_key: game.setups[board_key]}), name=name), load_stats(stats_file))
        for (_, name), (puzzle_file, stats_file) in zip(_SIDES, files)
    )
    return lower, upper
