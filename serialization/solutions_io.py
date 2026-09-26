import json
from collections.abc import Mapping
from pathlib import Path

from classes import Puzzle, Solution, SolutionBook
from .jsonio import block_grids_to_rows, dump_json, rows_to_block_grids


def single_entry(book, flat: bool, label: str):
    """For a `flat` save: the one entry of `book`, raising if there isn't
    exactly one. Returns None when not flat."""
    if not flat:
        return None
    if len(book) != 1:
        raise ValueError(
            f"flat=True requires exactly one puzzle in the {label}, got {len(book)}. "
            "Use flat=False for a multi-puzzle save."
        )
    return next(iter(book.values()))


def write_solutions(solution_book: SolutionBook, file_path: Path, flat: bool = True) -> None:
    """A solutions.json holds nothing but grids, written as letters (one
    string per row) so it can be read by eye. The puzzles they solve are
    never written: the run folder's path points at them in games/ (see
    Source.relative_dir()).

    `flat=True` (one puzzle's run, from solving.solve_puzzle -- its name
    is the folder above the run) writes just the list of grids;
    `flat=False` (a whole book, from solving.solve_puzzlebook) writes a
    list of `{"puzzle_name", "grids"}` objects."""
    sol = single_entry(solution_book, flat, "SolutionBook")
    if flat:
        data = _letter_grids(sol)
    else:
        data = [
            {"puzzle_name": sol.puzzle_name, "grids": _letter_grids(sol)}
            for sol in solution_book.values()
        ]
    dump_json(data, file_path)


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


def load_solutionbook(
    file_path: Path, puzzles: Mapping[str, Puzzle], flat_name: str, book_name: str = None
) -> SolutionBook:
    """Inverse of write_solutions: each puzzle's grids attached to its
    Puzzle from `puzzles`; a flat file's belong to `flat_name`. A grid is a
    list, never a dict, so the first element tells the layouts apart. Every
    grid is checked to be a solution (see Setup.grids_to_rows)."""
    with open(file_path, "r") as f:
        data = json.load(f)

    if data and isinstance(data[0], dict):
        entries = [(item["puzzle_name"], item["grids"]) for item in data]
    else:
        entries = [(flat_name, data)]

    solutions = []
    for name, rows in entries:
        if name not in puzzles:
            raise ValueError(f"{file_path} has solutions for puzzle {name!r}, which games/ doesn't have.")
        puzzle = puzzles[name]
        grids = rows_to_block_grids(rows, puzzle.blocks, puzzle.board.cells.shape)
        solutions.append(Solution(puzzle, puzzle.setup.grids_to_rows(grids)))
    return SolutionBook(*solutions, book_name=book_name)
