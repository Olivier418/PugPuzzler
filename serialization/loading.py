import json
from pathlib import Path

import numpy as np
from colorama import Back, Fore, Style

from classes import Setup, Game, Puzzle, PuzzleBook, Block, BlockCollection, Board, RegularBoard, PyramidBoard
from classes.source import Source
from classes.solutions import PuzzleInfo
from constants import GAMES_DIR
from .jsonio import parse_letter_grid, letter_grid_to_rows


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
        if "cells" in data:
            return PyramidBoard(cells=_cells_from_json(data["cells"]))
        # Default PyramidBoard construction using base dimensions
        width = data.get("width", 5)
        depth = data.get("height", 5)
        return PyramidBoard(width, depth)

    raise ValueError(f"Unsupported board type: '{board_type}'")


def grid_to_letter_rows(setup: Setup, grid: np.ndarray, empty: str = " ") -> list:
    """Inverse of Puzzle._initialize_grid: numeric grid -> one string per
    row (a list of those per layer on a pyramid).

    `grid` is compact (see Setup.__init__); expanded to the full board
    shape here since the JSON 'grid' convention is full-shape, human-
    authored rows."""
    full_grid = setup.expand(grid)
    letter_arr = np.full(full_grid.shape, empty, dtype="<U1")
    for idx, block in setup.blocks.items():
        letter_arr[full_grid == idx] = block.letter
    return letter_grid_to_rows(letter_arr)  # also undoes the transpose applied on load


def load_game(dir_path: Path | str) -> Game:
    """Auto-discover and load blocks, boards, puzzlebooks, and standalone puzzles."""
    dir_path = Path(dir_path)
    game_name = dir_path.name

    # 1. Load blocks
    blocks_path = dir_path / "blocks.json"
    if not blocks_path.exists():
        raise FileNotFoundError(f"Missing required blocks file: {blocks_path}")
    blocks = load_blocks(blocks_path)

    # 2. Load boards and create setups
    boards_path = dir_path / "boards.json"
    if not boards_path.exists():
        raise FileNotFoundError(f"Missing required boards file: {boards_path}")
    boards = load_boards(boards_path)

    setups = {board_key: Setup(blocks, board) for board_key, board in boards.items()}

    # 3. Discover PuzzleBooks from books/
    books_dir = dir_path / "books"
    loaded_books = []
    if books_dir.is_dir():
        for json_file in books_dir.glob("*.json"):
            with open(json_file, "r") as f:
                content = json.load(f)

            if isinstance(content, dict):
                book_name = json_file.stem
                board_key = content["board"]
                puzzle_data = content["puzzles"]
            else:
                book_name = json_file.stem
                board_key = "main"
                puzzle_data = content

            setup = setups[board_key]
            puzzles = [
                Puzzle(
                    setup,
                    parse_letter_grid(item["grid"]) if "grid" in item else None,
                    name=item["name"],
                    difficulty=item.get("difficulty"),
                )
                for item in puzzle_data
            ]
            book = PuzzleBook(*puzzles, name=book_name)

            # Record where this book -- and each puzzle inside it -- came
            # from, so Solution.from_puzzle/from_puzzlebook can later save
            # results to a mirrored solutions/ path automatically.
            book.source = Source(game_name=game_name, book_name=book_name)
            for p in puzzles:
                p.source = book.source.for_puzzle(p.name)

            loaded_books.append(book)

    # 4. Discover standalone puzzles from puzzles/
    puzzles_dir = dir_path / "puzzles"
    loaded_puzzles = []
    if puzzles_dir.is_dir():
        for json_file in puzzles_dir.glob("*.json"):
            with open(json_file, "r") as f:
                item = json.load(f)

            board_key = item.get("board", "main")
            setup = setups[board_key]
            grid = parse_letter_grid(item["grid"]) if "grid" in item else None
            name = json_file.stem

            puzzle = Puzzle(
                setup,
                letter_grid=grid,
                name=name,
                difficulty=item.get("difficulty"),
            )
            puzzle.source = Source(game_name=game_name, puzzle_name=name)
            loaded_puzzles.append(puzzle)

    return Game(
        books=loaded_books,
        puzzles=loaded_puzzles,
        setups=setups,
        name=game_name,
    )


def load_setup_for_puzzle(
    game_name: str,
    book_name: str = None,
    puzzle_name: str = None,
    games_root: str | Path = GAMES_DIR,
) -> Setup:
    """Builds Setup on-demand from canonical game/book/puzzle IDs.

    Only used as a cold-start fallback -- see Solution.to_states -- for
    when a Solution is loaded back from a bare solutions.json with no
    live Puzzle/PuzzleBook (and thus no live Setup) in memory.
    """
    game_dir = Path(games_root) / game_name

    # 1. Load block collection
    blocks = load_blocks(game_dir / "blocks.json")

    # 2. Determine target board key
    if book_name:
        target_file = game_dir / "books" / f"{book_name}.json"
        with open(target_file, "r") as f:
            content = json.load(f)
        board_key = content.get("board", "main") if isinstance(content, dict) else "main"
    elif puzzle_name:
        target_file = game_dir / "puzzles" / f"{puzzle_name}.json"
        with open(target_file, "r") as f:
            content = json.load(f)
        board_key = content.get("board", "main")
    else:
        board_key = "main"

    # 3. Load board config and return Setup
    boards = load_boards(game_dir / "boards.json")
    return Setup(blocks, boards[board_key])


def load_puzzle_info_for_puzzle(
    game_name: str,
    puzzle_name: str,
    book_name: str = None,
    games_root: str | Path = GAMES_DIR,
) -> PuzzleInfo:
    """Reads a puzzle's difficulty and empty-cell count straight off its
    JSON entry, without building a Setup or Puzzle.

    Unlike load_setup_for_puzzle, neither value needs the placements
    Setup exists to compute -- difficulty is just a JSON field, and the
    empty-cell count only needs the raw letter grid (or, if a puzzle has
    none, the board's cell count from boards.json, which is cheap on its
    own -- it's specifically placement computation that's expensive).

    Only used as a cold-start fallback -- see Solution.puzzle_info -- for
    when a Solution has no live Puzzle/PuzzleBook in memory to read
    difficulty/nr_empty_spaces off of directly.
    """
    game_dir = Path(games_root) / game_name

    if book_name:
        with open(game_dir / "books" / f"{book_name}.json", "r") as f:
            content = json.load(f)
        board_key = content.get("board", "main") if isinstance(content, dict) else "main"
        puzzle_data = content["puzzles"] if isinstance(content, dict) else content
        item = next(p for p in puzzle_data if p["name"] == puzzle_name)
    else:
        with open(game_dir / "puzzles" / f"{puzzle_name}.json", "r") as f:
            item = json.load(f)
        board_key = item.get("board", "main")

    difficulty = item.get("difficulty")

    if "grid" in item:
        grid = parse_letter_grid(item["grid"])
        nr_empty_spaces = int((grid == " ").sum())
    else:
        boards = load_boards(game_dir / "boards.json")
        nr_empty_spaces = int(boards[board_key].cells.sum())

    return PuzzleInfo(difficulty=difficulty, nr_empty_spaces=nr_empty_spaces)