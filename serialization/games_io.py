"""The games/ folder: a game's blocks, boards, books and standalone puzzles,
read into a Game -- and the one thing ever written there, a new book
(save_puzzlebook).

  games/<Game>/blocks.json         {block name: {"letter", "positions", "terminal"}}
  games/<Game>/boards.json         {board key: {"type", "width"/"depth"/"height" or "cells"}}
  games/<Game>/books/<book>.json   {"board": key, "puzzles": [{"name", "difficulty", "grid"}, ...]}
  games/<Game>/puzzles/<name>.json {"board": key, "difficulty", "grid"}

A puzzle file in that last format can also live anywhere else and be read
with load_puzzle onto a loaded Game (e.g. tutorial/demo_puzzle.json).

A grid is letter rows (see jsonio.parse_letter_grid); "difficulty" and
"grid" are optional. Names are never read from a JSON "name" field except
for puzzles inside a book file: Game.name is the folder's, a PuzzleBook's
and a standalone Puzzle's the file stem."""
import json
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path

from colorama import Back, Fore, Style

from classes import (
    Block, BlockCollection, Board, Game, Puzzle, PuzzleBook, PyramidBoard, RegularBoard, Setup, Source,
)
from constants import GAMES_DIR
from .jsonio import block_grids_to_rows, dump_json, parse_letter_grid


def load_blocks(json_path: Path | str) -> BlockCollection:
    """Read a blocks.json file into a BlockCollection."""
    with open(json_path, "r") as f:
        data = json.load(f)

    blocks = []
    for info in data.values():
        term = info["terminal"]
        terminal_color = getattr(Back, term["bg"]) + getattr(Fore, term["fg"])
        if "style" in term:
            terminal_color += getattr(Style, term["style"])
        blocks.append(Block(positions=info["positions"], letter=info["letter"], terminal_color=terminal_color))
    return BlockCollection(*blocks)


def _parse_board(data: dict) -> Board:
    """Instantiate a Board subclass from one boards.json entry."""
    board_type = data["type"]

    if board_type == "RegularBoard":
        if "cells" in data:
            # Authored like a puzzle's letter grid, one row string per depth
            # position ("X" = cell), i.e. (depth, width) as written; the
            # internal convention is (width, depth), so it gets the same
            # one-time transpose Puzzle._initialize_grid gives a letter grid.
            return RegularBoard(cells=(parse_letter_grid(data["cells"]) == "X").T)
        return RegularBoard(width=data["width"], depth=data["depth"], height=data.get("height"))

    if board_type == "PyramidBoard":
        return PyramidBoard(width=data["width"], depth=data["depth"])

    raise ValueError(f"Unsupported board type: '{board_type}'")


def load_boards(json_path: Path | str) -> dict[str, Board]:
    """Read a boards.json file into {board_key: Board}."""
    with open(json_path, "r") as f:
        return {key: _parse_board(info) for key, info in json.load(f).items()}


def _required(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required game file: {path}")
    return path


@lru_cache(maxsize=None)
def _setups(game_dir: Path) -> dict[str, Setup]:
    blocks = load_blocks(_required(game_dir / "blocks.json"))
    boards = load_boards(_required(game_dir / "boards.json"))
    return {key: Setup(blocks, board) for key, board in boards.items()}


def load_setups(game_dir: Path | str) -> dict[str, Setup]:
    """{board_key: Setup} for every board of a game folder, all over the
    game's one BlockCollection.

    Built once per game folder per process: everything loaded from a game
    (load_game, results_io.load_run_books, ...) shares the same Setup
    objects, so a puzzle loaded with a run and the Game you loaded yourself
    are on the very same Setup -- which Game.board_key, and so save_bounds /
    save_puzzlebook, match by identity. games/ never changes, so the cache
    can't go stale."""
    return dict(_setups(Path(game_dir).resolve()))


# ---- puzzles and books ------------------------------------------------------

def _setup_for(setups: dict[str, Setup], entry: dict, json_file: Path) -> Setup:
    board_key = entry["board"]
    if board_key not in setups:
        raise ValueError(f"{json_file} is on board {board_key!r}, which the game lacks (it has {sorted(setups)}).")
    return setups[board_key]


def _puzzle_from_entry(setup: Setup, entry: dict, name: str) -> Puzzle:
    """One puzzle from its JSON entry (a book's list item or a standalone
    puzzle file). Building it checks every piece is a legal placement."""
    return Puzzle(
        setup,
        parse_letter_grid(entry["grid"]) if "grid" in entry else None,
        name=name,
        difficulty=entry.get("difficulty"),
    )


def load_puzzles(json_file: Path | str, setups: dict[str, Setup]) -> list[Puzzle]:
    """The puzzles of a book file, built on the Setup of its board."""
    with open(json_file, "r") as f:
        content = json.load(f)
    setup = _setup_for(setups, content, json_file)
    return [_puzzle_from_entry(setup, entry, entry["name"]) for entry in content["puzzles"]]


def write_puzzles(puzzles: Iterable[Puzzle], board_key: str, json_file: Path | str) -> None:
    """Inverse of load_puzzles, for puzzles on the board `board_key`."""
    entries = []
    for p in puzzles:
        entry = {"name": p.name}
        if p.difficulty is not None:
            entry["difficulty"] = p.difficulty
        entry["grid"] = block_grids_to_rows(p.setup.to_full_grid(p.grid)[None], p.blocks)[0]
        entries.append(entry)
    dump_json({"board": board_key, "puzzles": entries}, json_file)


# Every book and puzzle loaded below records where it came from (its
# Source), so solving.solve_puzzle/solve_puzzlebook can save results to a
# mirrored solutions/ path, and a saved run can find its puzzles again.

def _set_book_source(book: PuzzleBook, game_name: str) -> None:
    book.source = Source(game_name=game_name, book_name=book.name)
    for puzzle in book.values():
        puzzle.source = Source(game_name=game_name, book_name=book.name, puzzle_name=puzzle.name)


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
        puzzle = _puzzle_from_entry(_setup_for(setups, entry, json_file), entry, json_file.stem)
        puzzle.source = Source(game_name=game_dir.name, puzzle_name=puzzle.name)
        puzzles.append(puzzle)
    return puzzles


def load_puzzle(json_file: Path | str, game: Game) -> Puzzle:
    """A puzzle file kept outside games/ (same format as games/<game>/puzzles/*.json),
    on `game`'s Setup for its board and named after the file. It has no Source,
    so it can be solved and shown but its runs can't be saved."""
    json_file = Path(json_file)
    with open(json_file, "r") as f:
        entry = json.load(f)
    return _puzzle_from_entry(_setup_for(game.setups, entry, json_file), entry, json_file.stem)


def load_game(dir_path: Path | str) -> Game:
    """Auto-discover and load blocks, boards, puzzlebooks, and standalone puzzles."""
    dir_path = Path(dir_path)
    setups = load_setups(dir_path)
    books = [load_book(dir_path, f.stem, setups) for f in sorted((dir_path / "books").glob("*.json"))]
    return Game(dir_path.name, setups, books, load_standalone_puzzles(dir_path, setups))


def save_puzzlebook(book: PuzzleBook, game: Game, games_root: Path | str = GAMES_DIR) -> Path:
    """Give a book made in memory (e.g. by compute_puzzle_bounds) a home
    in `game`: write it to games/<game>/books/<book.name>.json, set its
    Source and add it to `game.books` -- after which it is solved, saved
    and loaded back like any other book. Never overwrites a book."""
    board_key = game.board_key(book.setup)
    json_file = Path(games_root) / game.name / "books" / f"{book.name}.json"
    if json_file.exists() or book.name in game.books:
        raise FileExistsError(f"Game {game.name!r} already has a book {book.name!r} ({json_file}).")
    json_file.parent.mkdir(parents=True, exist_ok=True)
    write_puzzles(book.values(), board_key, json_file)
    _set_book_source(book, game.name)
    game.books[book.name] = book
    return json_file
