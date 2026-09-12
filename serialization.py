import json
from pathlib import Path
import numpy as np
from colorama import Back, Fore, Style

from classes import (
    Block,
    BlockCollection,
    Puzzle,
    PuzzleBook,
    PuzzleSetup,
    SolutionBook,
    SolvedPuzzle,
)


# 1. Helper function to load blocks
def load_block_collection(json_path: Path | str) -> BlockCollection:
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


# 2. Reusable puzzle loader taking an already-built setup
def load_puzzles(file_path: Path | str, setup: PuzzleSetup) -> PuzzleBook:
    with open(file_path, "r") as f:
        data = json.load(f)

    # setup (board + blocks + the expensive valid-placement search) is built
    # once by the caller and shared by every Puzzle below -- and, via
    # PuzzleBook.setup, by anyone loading this book's solutions later too.
    puzzles = [
        Puzzle(
            setup,
            np.array(item["grid"], dtype=str),
            name=item["name"],
            difficulty=item["difficulty"],
        )
        for item in data
    ]
    return PuzzleBook(*puzzles)


def grid_to_letter_rows(setup: PuzzleSetup, grid: np.ndarray, empty: str = " ") -> list[list[str]]:
    """Inverse of Puzzle._initialize_grid: numeric grid -> row-major letters."""
    letter_arr = np.full(grid.shape, empty, dtype="<U1")
    for idx, block in setup.blocks.items():
        letter_arr[grid == idx] = block.letter
    return letter_arr.T.tolist()  # undo the transpose applied on load


def save_solutions(solution_book: SolutionBook, file_path: Path | str) -> None:
    """Persist an already-computed SolutionBook. Pure serialization: does
    no solving -- build the SolutionBook first via
    SolutionBook.from_puzzlebook(...)."""
    grouped = [
        {
            "name": sp.puzzle.name,
            "difficulty": sp.puzzle.difficulty,
            "puzzle": grid_to_letter_rows(sp.puzzle.setup, sp.puzzle.grid),
            "solutions": [grid_to_letter_rows(sol.setup, sol.grid) for sol in sp.solutions],
        }
        for sp in solution_book.values()
    ]
    with open(file_path, "w") as f:
        json.dump(grouped, f, indent=2)


def load_solutions(file_path: Path | str, setup: PuzzleSetup) -> SolutionBook:
    """Recover a previously-saved SolutionBook without re-solving anything.

    Pass the setup you already have -- e.g. `main_puzzle_book.setup` -- so
    the (expensive) placement search isn't redone just to load solutions
    for a board you've already set up.
    """
    with open(file_path, "r") as f:
        data = json.load(f)

    solved_puzzles = [
        SolvedPuzzle(
            puzzle=Puzzle(
                setup,
                np.array(item["puzzle"], dtype=str),
                name=item["name"],
                difficulty=item["difficulty"],
            ),
            solutions=[
                Puzzle(
                    setup,
                    np.array(sol_grid, dtype=str),
                    name=f"{item['name']} (solution {i + 1})",
                    difficulty=item["difficulty"],
                )
                for i, sol_grid in enumerate(item["solutions"])
            ],
        )
        for item in data
    ]
    return SolutionBook(*solved_puzzles)