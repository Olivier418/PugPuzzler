"""Shared helpers for the test suite. Read-only: game definitions (games/) and
the known-solution fixtures (tests/data/) are the only files touched."""
import json
from pathlib import Path

import numpy as np

from classes import Puzzle
from constants import EMPTY, GAMES_DIR, UNPLACED
from serialization import load_game

ROOT = Path(__file__).resolve().parent.parent
KNOWN_SOLUTIONS_DIR = ROOT / "tests" / "data"


def load(game_name: str):
    return load_game(str(ROOT / GAMES_DIR / game_name))


def load_known_solutions(game_name: str) -> dict:
    """Hand-checked solutions from tests/data/<game>/<book>.json (a test
    fixture, unrelated to the solver's own solutions/ output) as
    {(book, puzzle_name): [solution, ...]}. A solution is a list of layers
    (one on a main/alt board, five on a pyramid) and a layer a list of row
    strings, " " marking a cell that is not on the board."""
    known = {}
    for path in sorted((KNOWN_SOLUTIONS_DIR / game_name).glob("*.json")):
        with open(path, encoding="utf-8") as f:
            for name, solutions in json.load(f).items():
                known[(path.stem, name)] = solutions
    return known


def assert_valid_solution(test, puzzle, state):
    """`state` (a fully placed State) is a legal solution to `puzzle`."""
    test.assertFalse((state.grid == EMPTY).any(), "empty cell left")
    for idx, p in state.chosen_placement_idx.items():
        test.assertNotEqual(p, UNPLACED, f"block {idx} not placed")
        cells = state.placement_cells[idx][p]
        test.assertTrue((state.grid[cells] == idx).all(), f"block {idx} grid/placement mismatch")
    # every cell is covered by exactly one block's placement
    covered = np.concatenate([state.placement_cells[i][p] for i, p in state.chosen_placement_idx.items()])
    test.assertEqual(len(covered), len(set(covered.tolist())), "blocks overlap")
    test.assertEqual(len(covered), state.setup.n_cells, "board not fully covered")
    # the puzzle's pre-filled cells are untouched
    given = puzzle.grid != EMPTY
    test.assertTrue((state.grid[given] == puzzle.grid[given]).all(), "pre-filled cells changed")


def unsolvable_puzzle(game) -> Puzzle:
    """Every piece is placed except F (3 cells) and H (5 cells); the 8 empty
    cells (blank below) form a 3x3 block whose top-middle cell is taken by D,
    i.e. a "U" opening upwards. F fits inside it, and so does H, but never
    both at once."""
    setup = game.books["main_puzzles"]["1"].setup
    letters = [
        "E", "E", "G", "G", "G", "J", "J", "J", "J", "I", "I",
        "A", "E", "E", "E", "G", "C", "D", "D", "D", "D", "I",
        "A", "A", "A", "L", "G", "C", " ", "D", " ", "I", "I",
        "B", "B", "L", "L", "L", "C", " ", " ", " ", "K", "K",
        "B", "B", "B", "L", "C", "C", " ", " ", " ", "K", "K",
    ]
    return Puzzle(setup, np.array(letters).reshape(5, 11), name="unsolvable")
