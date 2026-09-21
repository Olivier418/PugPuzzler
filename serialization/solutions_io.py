import json
from pathlib import Path

import numpy as np

from classes import Solution, SolutionBook
from constants import GAMES_DIR, OUTSIDE_BOARD
from .jsonio import dump_json, letter_grid_to_rows, parse_letter_grid
from .loading import load_blocks
from .paths import game_book_from_run_dir, puzzle_name_from_run_dir, single_entry


def write_solutions(solution_book: SolutionBook, file_path: Path, flat: bool = True) -> None:
    """game_name/book_name/puzzle_name are never written here -- they're
    already fully determined by the folder this file is saved under (see
    Source.relative_dir()), so storing them again in the JSON would just
    be a second copy of the same strings that could drift from the path.

    A solutions.json holds nothing but grids (or, for a multi-puzzle
    save, puzzle_name+grids pairs), written as letters -- one string per
    row, see _grids_to_rows -- so it can be read by eye. There's no
    sibling metadata to justify wrapping that in an object, so the file's
    top level is the list itself.

    `flat=True` (a single puzzle's solutions -- from solving.solve_puzzle,
    whether standalone or one puzzle solved out of a book) writes just
    the list of grids, since the puzzle name is recovered from the
    containing folder on load instead. `flat=False` (a whole book solved
    at once, from solving.solve_puzzlebook) always writes a list of
    `{"puzzle_name", "grids"}` objects, even for a one-puzzle book, since
    a batch save's folder doesn't carry a per-puzzle name the way a
    single-puzzle save's does.
    """
    sol = single_entry(solution_book, flat, "SolutionBook")
    if flat:
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
    only from the folder the file sits in (see game_book_from_run_dir)."""
    if game_name is None:
        raise ValueError(
            "Cannot tell which game's blocks the letters in this solutions file refer to: "
            "its folder doesn't follow <game>/(books|puzzles)/..."
        )
    return load_blocks(Path(GAMES_DIR) / game_name / "blocks.json")


def load_solutionbook(path: str | Path) -> SolutionBook:
    """Load a solutions.json (or the run folder holding one). The file's
    top level is a bare list (see write_solutions): a list of
    `{"puzzle_name", "grids"}` objects for a multi-puzzle save, or just a
    list of grids for a single-puzzle one. A grid is itself a list of
    lists, never a dict, so which shape it is can be told apart just by
    looking at the first element -- an empty list (no solutions found) is
    ambiguous, but defaults to the far more common single-puzzle case."""
    path = Path(path)
    file_path = path / "solutions.json" if path.is_dir() else path

    with open(file_path, "r") as f:
        data = json.load(f)

    game_name, book_name = game_book_from_run_dir(file_path)

    if data and isinstance(data[0], dict):
        entries = [(item["puzzle_name"], item["grids"]) for item in data]
    else:
        entries = [(puzzle_name_from_run_dir(file_path), data)]

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
    return SolutionBook(*solutions, game_name=game_name, book_name=book_name)
