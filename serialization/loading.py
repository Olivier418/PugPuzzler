import json
from pathlib import Path

import numpy as np
from colorama import Back, Fore, Style

from classes import (
    Block, BlockCollection, Board, Game, Puzzle, PuzzleBook, PyramidBoard, RegularBoard, Setup, Source,
)
from constants import GAMES_DIR
from .jsonio import parse_letter_grid
from .puzzles_io import DEFAULT_BOARD, load_puzzles, puzzle_from_entry, setup_for, write_puzzles


def load_blocks(json_path: Path | str) -> BlockCollection:
    """Read a blocks.json file into a BlockCollection."""
    with open(json_path, "r") as f:
        data = json.load(f)

    blocks = []
    for name, info in data.items():
        term = info["terminal"]
        terminal_color = getattr(Back, term["bg"]) + getattr(Fore, term["fg"])
        if "style" in term:
            terminal_color += getattr(Style, term["style"])

        blocks.append(
            Block(
                positions=np.array(info["positions"], dtype=bool),
                rgb=tuple(info["rgb"]),
                letter=info["letter"],
                terminal_color=terminal_color,
            )
        )
    return BlockCollection(*blocks)


def load_boards(json_path: Path | str) -> dict[str, Board]:
    """Read a boards.json file into {board_key: Board}."""
    with open(json_path, "r") as f:
        boards_config = json.load(f)
    return {key: _parse_board(info) for key, info in boards_config.items()}


def _cells_from_json(raw_cells: list) -> np.ndarray:
    """Convert a boards.json 'cells' field into this codebase's internal
    cells.shape == (width, depth) convention (see classes.rendering.grid_lines,
    which unpacks width, depth = shape[0], shape[1]).

    'cells' is authored the same human-readable way as a Puzzle letter
    grid: one row string per depth position ("X" = board cell, " " = not),
    `width` characters per row -- i.e. literally shape (depth, width) as
    written. Puzzle._initialize_grid
    already transposes a letter grid for exactly this reason; board cells
    need the identical, one-time transpose here, at the JSON boundary,
    rather than compensating for it downstream.
    """
    return (parse_letter_grid(raw_cells) == "X").T


def _parse_board(data: dict) -> Board:
    """Instantiate a Board subclass from one boards.json entry."""
    board_type = data.get("type", "Board")

    if board_type == "RegularBoard":
        if "cells" in data:
            return RegularBoard(cells=_cells_from_json(data["cells"]))
        elif "height" in data:
            return RegularBoard(width=data["width"], depth=data["depth"], height=data["height"])
        return RegularBoard(width=data["width"], depth=data["depth"])

    if board_type == "PyramidBoard":
        return PyramidBoard(width=data.get("width", 5), depth=data.get("depth", 5))

    raise ValueError(f"Unsupported board type: '{board_type}'")


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required game file: {path}")
    return path


def load_setups(game_dir: Path | str) -> dict[str, Setup]:
    """{board_key: Setup} for every board of a game folder, all over the
    game's one BlockCollection."""
    game_dir = Path(game_dir)
    blocks = load_blocks(_required(game_dir / "blocks.json"))
    boards = load_boards(_required(game_dir / "boards.json"))
    return {key: Setup(blocks, board) for key, board in boards.items()}


# Every book and puzzle loaded below records where it came from (its
# Source), so solving.solve_puzzle/solve_puzzlebook can save results to a
# mirrored solutions/ path, and a saved run can find its puzzles again.

def _set_book_source(book: PuzzleBook, game_name: str) -> None:
    book.source = Source(game_name=game_name, book_name=book.name)
    for puzzle in book.values():
        puzzle.source = book.source.for_puzzle(puzzle.name)


def load_book(game_dir: Path | str, book_name: str, setups: dict[str, Setup]) -> PuzzleBook:
    """The book games/<game>/books/<book_name>.json, on the game's `setups`."""
    game_dir = Path(game_dir)
    json_file = game_dir / "books" / f"{book_name}.json"
    if not json_file.exists():
        raise FileNotFoundError(f"Game {game_dir.name!r} has no book {book_name!r} ({json_file}).")
    book = PuzzleBook(*load_puzzles(json_file, setups), name=book_name)
    _set_book_source(book, game_dir.name)
    return book


def load_standalone_puzzles(game_dir: Path | str, setups: dict[str, Setup]) -> list[Puzzle]:
    """Every games/<game>/puzzles/*.json: one puzzle entry plus its board
    key per file, each puzzle named after its file."""
    game_dir = Path(game_dir)
    puzzles = []
    for json_file in sorted((game_dir / "puzzles").glob("*.json")):
        with open(json_file, "r") as f:
            entry = json.load(f)
        setup = setup_for(setups, entry.get("board", DEFAULT_BOARD), json_file)
        puzzle = puzzle_from_entry(setup, entry, json_file.stem)
        puzzle.source = Source(game_name=game_dir.name, puzzle_name=puzzle.name)
        puzzles.append(puzzle)
    return puzzles


def load_game(dir_path: Path | str) -> Game:
    """Auto-discover and load blocks, boards, puzzlebooks, and standalone puzzles."""
    dir_path = Path(dir_path)
    setups = load_setups(dir_path)
    books = [load_book(dir_path, f.stem, setups) for f in sorted((dir_path / "books").glob("*.json"))]
    puzzles = load_standalone_puzzles(dir_path, setups)
    return Game(books=books, puzzles=puzzles, setups=setups, name=dir_path.name)


def save_puzzlebook(book: PuzzleBook, game: Game, games_root: Path | str = GAMES_DIR) -> Path:
    """Give a book made in memory (e.g. by most_difficult_puzzles) a home
    in `game`: write it to games/<game>/books/<book.name>.json, set its
    Source and add it to `game.books` -- after which it is solved, saved
    and loaded back like any other book. Never overwrites a book."""
    board_key = next((key for key, setup in game.setups.items() if setup is book.setup), None)
    if board_key is None:
        raise ValueError(f"Book {book.name!r} isn't built on one of game {game.name!r}'s Setups.")
    json_file = Path(games_root) / game.name / "books" / f"{book.name}.json"
    if json_file.exists() or book.name in game.books:
        raise FileExistsError(f"Game {game.name!r} already has a book {book.name!r} ({json_file}).")
    json_file.parent.mkdir(parents=True, exist_ok=True)
    write_puzzles(book.values(), board_key, json_file)
    _set_book_source(book, game.name)
    game.books[book.name] = book
    return json_file
