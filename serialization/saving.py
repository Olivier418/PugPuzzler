import json
from pathlib import Path

import numpy as np

from classes import SolutionBook, Solution
from classes._utils import _next_free_idx_dir
from classes.solutions import SolveStats, SolveStatsBook
from constants import GAMES_DIR, OUTSIDE_BOARD
from .jsonio import dump_json, letter_grid_to_rows, parse_letter_grid
from .loading import load_blocks


def _write_solutions(solution_book: SolutionBook, file_path: Path, flat: bool = True) -> None:
    """game_name/book_name/puzzle_name are never written here -- they're
    already fully determined by the folder this file is saved under (see
    Source.relative_dir()), so storing them again in the JSON would just
    be a second copy of the same strings that could drift from the path.

    A solutions.json holds nothing but grids (or, for a multi-puzzle
    save, puzzle_name+grids pairs), written as letters -- one string per
    row, see _grids_to_rows -- so it can be read by eye. There's no sibling metadata to
    justify wrapping that in an object, so the file's top level is the
    list itself.

    `flat=True` (a single puzzle's solutions -- from Solution.from_puzzle,
    whether standalone or one puzzle solved out of a book) writes just
    the list of grids, since the puzzle name is recovered from the
    containing folder on load instead. `flat=False` (a whole book solved
    at once, from SolutionBook.from_puzzlebook) always writes a list of
    `{"puzzle_name", "grids"}` objects, even for a one-puzzle book, since
    a batch save's folder doesn't carry a per-puzzle name the way a
    single-puzzle save's does.
    """
    if flat:
        if len(solution_book) != 1:
            raise ValueError(
                f"flat=True requires exactly one puzzle in the SolutionBook, got {len(solution_book)}. "
                "Use flat=False for a multi-puzzle save."
            )
        sol = next(iter(solution_book.values()))
        data = _grids_to_rows(sol)
    else:
        data = [
            {"puzzle_name": sol.puzzle_name, "grids": _grids_to_rows(sol)}
            for sol in solution_book.values()
        ]
    dump_json(data, file_path)


def _grids_to_rows(sol: Solution) -> list:
    """A Solution's grids as letter rows (see letter_grid_to_rows): each
    block index becomes its block's letter, anything else (the cells
    outside a non-rectangular board or a pyramid's upper layers) a space.
    The blocks come from the live puzzle if there is one, else from the
    game's blocks.json."""
    if not sol.grids:
        return []
    blocks = sol.puzzle.blocks if sol.puzzle is not None else _game_blocks(sol.game_name)
    rows = []
    for grid in sol.grids:
        letters = np.full(grid.shape, " ", dtype="<U1")
        for idx, block in blocks.items():
            letters[grid == idx] = block.letter
        rows.append(letter_grid_to_rows(letters))
    return rows


def _rows_to_grid(rows: list, blocks) -> np.ndarray:
    """Inverse of _grids_to_rows for one grid: letters -> block indices,
    spaces -> OUTSIDE_BOARD."""
    letters = parse_letter_grid(rows).T
    grid = np.full(letters.shape, OUTSIDE_BOARD, dtype=int)
    for idx, block in blocks.items():
        grid[letters == block.letter] = idx
    return grid


def _game_blocks(game_name: str | None):
    """The blocks a solutions file's letters refer to. The game is known
    only from the folder the file sits in (see _game_book_from_run_dir)."""
    if game_name is None:
        raise ValueError(
            "Cannot tell which game's blocks the letters in this solutions file refer to: "
            "its folder doesn't follow <game>/(books|puzzles)/..."
        )
    return load_blocks(Path(GAMES_DIR) / game_name / "blocks.json")


def save_solutions(solution_book: SolutionBook, file_path: str | Path) -> Path:
    file_path = Path(file_path)
    flat = len(solution_book) == 1

    if file_path.suffix == ".json":
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_solutions(solution_book, file_path, flat=flat)
        return file_path

    name = solution_book.name or "solutions"
    folder = _next_free_idx_dir(file_path / name)
    folder.mkdir(parents=True, exist_ok=True)
    _write_solutions(solution_book, folder / "solutions.json", flat=flat)
    return folder


def _write_solve_stats(stats_book: SolveStatsBook, file_path: Path, flat: bool = True) -> None:
    """Mirrors _write_solutions: game_name/book_name/puzzle_name are
    never written, since the folder already determines them."""
    if flat:
        if len(stats_book) != 1:
            raise ValueError(
                f"flat=True requires exactly one puzzle in the SolveStatsBook, got {len(stats_book)}. "
                "Use flat=False for a multi-puzzle save."
            )
        stats = next(iter(stats_book.values()))
        data = {
            "options": stats_book.options,
            "seed": stats_book.seed,
            "duration": stats.duration,
            "elapsed": stats.elapsed,
        }
    else:
        data = {
            "options": stats_book.options,
            "seed": stats_book.seed,
            "puzzles": [
                {
                    "puzzle_name": stats.puzzle_name,
                    "options": stats.options,
                    "seed": stats.seed,
                    "duration": stats.duration,
                    "elapsed": stats.elapsed,
                }
                for stats in stats_book.values()
            ],
        }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def save_solve_stats(stats_book: SolveStatsBook, file_path: str | Path) -> Path:
    file_path = Path(file_path)
    flat = len(stats_book) == 1

    if file_path.suffix == ".json":
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_solve_stats(stats_book, file_path, flat=flat)
        return file_path

    name = stats_book.name or "stats"
    folder = _next_free_idx_dir(file_path / name)
    folder.mkdir(parents=True, exist_ok=True)
    _write_solve_stats(stats_book, folder / "stats.json", flat=flat)
    return folder


def save_solution_run(
    solution_book: SolutionBook, stats_book: SolveStatsBook, folder: str | Path, flat: bool = True
) -> Path:
    """Save a Solution/SolveStats pair produced by the same solve into
    one run folder, as solutions.json + stats.json side by side.

    `flat=True` (the default) drops the "puzzles" wrapper for a
    single-puzzle result (Solution.from_puzzle, standalone or out of a
    book) -- puzzle_name is recovered on load from `folder` itself
    (`.../<puzzle_name>/<idx>`), same as game_name/book_name are from the
    rest of the path (see Source.relative_dir()). Callers saving more
    than one puzzle's results into one file -- a whole book solved via
    SolutionBook.from_puzzlebook, or plotting.benchmark's per-trial
    folders that don't follow the `<puzzle_name>/<idx>` convention --
    must pass `flat=False` to keep puzzle_name explicit in the JSON.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    _write_solutions(solution_book, folder / "solutions.json", flat=flat)
    _write_solve_stats(stats_book, folder / "stats.json", flat=flat)
    return folder


def _puzzle_name_from_run_dir(file_path: Path) -> str:
    """Recover a flattened file's puzzle_name from its folder, per the
    convention every writer here uses: .../<puzzle_name>/<idx>/<file>."""
    return file_path.parent.parent.name


def _game_book_from_run_dir(file_path: Path) -> tuple[str | None, str | None]:
    """Best-effort recovery of (game_name, book_name) from a run folder
    following the <game>/(books/<book>[/<puzzle>]|puzzles/<puzzle>)/<idx>
    convention every solutions/ writer here uses -- the inverse of
    Source.relative_dir(). Returns (None, None) for a folder that
    doesn't follow it (e.g. plotting.benchmark's nested config/seed
    layout), where callers already have puzzle_name explicit in the
    JSON and don't need this."""
    parts = file_path.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "puzzles" and i > 0:
            return parts[i - 1], None
        if parts[i] == "books" and i > 0 and i + 1 < len(parts):
            return parts[i - 1], parts[i + 1]
    return None, None


def _read_solutions(path: str | Path) -> tuple[list[Solution], dict]:
    """Parse a solutions.json into its Solutions plus the book-level
    metadata (game_name, book_name) -- the one place that reads the
    file, shared by load_solutionbook and load_solution so neither has
    to go through the other's container type.

    The file's top level is a bare list (see _write_solutions): a list
    of `{"puzzle_name", "grids"}` objects for a multi-puzzle save, or
    just a list of grids for a single-puzzle one. A grid is itself a
    list of lists, never a dict, so which shape it is can be told apart
    just by looking at the first element -- an empty list (no solutions
    found) is ambiguous, but defaults to the far more common single-
    puzzle case."""
    path = Path(path)
    file_path = path / "solutions.json" if path.is_dir() else path

    with open(file_path, "r") as f:
        data = json.load(f)

    game_name, book_name = _game_book_from_run_dir(file_path)

    if data and isinstance(data[0], dict):
        entries = [(item["puzzle_name"], item["grids"]) for item in data]
    else:
        entries = [(_puzzle_name_from_run_dir(file_path), data)]

    blocks = _game_blocks(game_name) if any(rows for _, rows in entries) else None
    solutions = [
        Solution(
            puzzle_name=puzzle_name,
            grids=[_rows_to_grid(grid, blocks) for grid in grids],
            game_name=game_name,
            book_name=book_name,
        )
        for puzzle_name, grids in entries
    ]

    meta = {
        "game_name": game_name,
        "book_name": book_name,
    }
    return solutions, meta


def load_solutionbook(path: str | Path) -> SolutionBook:
    solutions, meta = _read_solutions(path)
    return SolutionBook(*solutions, **meta)


def load_solution(path: str | Path) -> Solution:
    """Load a solutions file saved from a single puzzle (see
    Solution.from_puzzle) and return the bare Solution, instead of
    making the caller index a one-entry SolutionBook themselves.

    Raises if the file holds more than one puzzle's solutions -- use
    load_solutionbook() for those instead.
    """
    solutions, _ = _read_solutions(path)
    if len(solutions) != 1:
        names = ", ".join(sol.puzzle_name for sol in solutions)
        raise ValueError(
            f"Expected a single-puzzle solutions file, found {len(solutions)} puzzles "
            f"({names}) in {path}. Use load_solutionbook() instead."
        )
    return solutions[0]


# Solver flags that older stats.json files recorded as top-level keys before
# they became the generic "options" dict; folded back into it on load so old
# runs (and benchmark folders) still load, and still tell configs apart.
_LEGACY_OPTION_KEYS = ("partition_pruning", "partition_branching", "symmetry_branching")


def _read_options(data: dict, default: dict | None = None) -> dict:
    """The solver options recorded in a stats.json (or one puzzle's entry
    in it): the "options" dict if present, else any legacy flag keys, else
    `default` (the book-level value, for a per-puzzle entry that has none
    of its own)."""
    if "options" in data:
        return dict(data["options"])
    legacy = {k: data[k] for k in _LEGACY_OPTION_KEYS if k in data}
    if legacy:
        return legacy
    return dict(default or {})


def _read_solve_stats(path: str | Path) -> tuple[list[SolveStats], dict]:
    """Parse a stats.json into its SolveStats plus the book-level
    metadata -- mirrors _read_solutions."""
    path = Path(path)
    file_path = path / "stats.json" if path.is_dir() else path

    with open(file_path, "r") as f:
        data = json.load(f)

    game_name, book_name = _game_book_from_run_dir(file_path)
    book_options = _read_options(data)
    book_seed = data.get("seed")

    if "puzzles" in data:
        stats = [
            SolveStats(
                puzzle_name=item["puzzle_name"],
                options=_read_options(item, default=book_options),
                seed=item.get("seed", book_seed),
                duration=item["duration"],
                elapsed=item["elapsed"],
            )
            for item in data["puzzles"]
        ]
    else:
        stats = [
            SolveStats(
                puzzle_name=_puzzle_name_from_run_dir(file_path),
                options=book_options,
                seed=book_seed,
                duration=data["duration"],
                elapsed=data["elapsed"],
            )
        ]

    meta = {
        "game_name": game_name,
        "book_name": book_name,
        "options": book_options,
        "seed": book_seed,
    }
    return stats, meta


def load_solve_stats_book(path: str | Path) -> SolveStatsBook:
    stats, meta = _read_solve_stats(path)
    return SolveStatsBook(*stats, **meta)


def load_solve_stats(path: str | Path) -> SolveStats:
    """Load a stats file saved from a single puzzle and return the bare
    SolveStats, instead of making the caller index a one-entry
    SolveStatsBook themselves. Raises if the file holds more than one
    puzzle's stats -- use load_solve_stats_book() for those instead."""
    stats, _ = _read_solve_stats(path)
    if len(stats) != 1:
        names = ", ".join(s.puzzle_name for s in stats)
        raise ValueError(
            f"Expected a single-puzzle stats file, found {len(stats)} puzzles "
            f"({names}) in {path}. Use load_solve_stats_book() instead."
        )
    return stats[0]


def load_solution_run(folder: str | Path) -> tuple[SolutionBook, SolveStatsBook]:
    """Load a run folder saved by save_solution_run (or
    Solution.from_puzzle/SolutionBook.from_puzzlebook with save=True)
    back into its (SolutionBook, SolveStatsBook) pair."""
    folder = Path(folder)
    return load_solutionbook(folder / "solutions.json"), load_solve_stats_book(folder / "stats.json")
