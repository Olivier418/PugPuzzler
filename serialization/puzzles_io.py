"""The book-file format: one board key plus a list of named puzzles, each an
optional letter grid and difficulty --
`{"board": key, "puzzles": [{"name", "difficulty", "grid"}, ...]}`, or, for
the "main" board, just the bare list. games/<Game>/books/*.json are written
in it."""
import json
from collections.abc import Iterable
from pathlib import Path

from classes import Puzzle, Setup
from .jsonio import block_grid_to_rows, dump_json, parse_letter_grid

DEFAULT_BOARD = "main"


def read_book_file(json_file: Path | str) -> tuple[str, list[dict]]:
    """(board_key, puzzle entries) of a book-format file."""
    with open(json_file, "r") as f:
        content = json.load(f)
    if isinstance(content, dict):
        return content.get("board", DEFAULT_BOARD), content["puzzles"]
    return DEFAULT_BOARD, content


def puzzle_from_entry(setup: Setup, entry: dict, name: str) -> Puzzle:
    """One puzzle from its JSON entry (a book's list item or a standalone
    puzzle file). Building it checks every piece is a legal placement."""
    return Puzzle(
        setup,
        parse_letter_grid(entry["grid"]) if "grid" in entry else None,
        name=name,
        difficulty=entry.get("difficulty"),
    )


def setup_for(setups: dict[str, Setup], board_key: str, json_file: Path | str) -> Setup:
    if board_key not in setups:
        raise ValueError(f"{json_file} is on board {board_key!r}, which the game lacks (it has {sorted(setups)}).")
    return setups[board_key]


def load_puzzles(json_file: Path | str, setups: dict[str, Setup]) -> list[Puzzle]:
    """The puzzles of a book-format file, built on `setups[board_key]`."""
    board_key, entries = read_book_file(json_file)
    setup = setup_for(setups, board_key, json_file)
    return [puzzle_from_entry(setup, entry, entry["name"]) for entry in entries]


def write_puzzles(puzzles: Iterable[Puzzle], board_key: str, json_file: Path | str) -> None:
    """Inverse of load_puzzles, for puzzles on the board `board_key`."""
    entries = []
    for p in puzzles:
        entry = {"name": p.name}
        if p.difficulty is not None:
            entry["difficulty"] = p.difficulty
        entry["grid"] = block_grid_to_rows(p.setup.to_full_grid(p.grid), p.blocks)
        entries.append(entry)
    dump_json({"board": board_key, "puzzles": entries}, json_file)
