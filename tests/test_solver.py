"""Solver tests: solution counts known from outside this project, and the
solver's options, which may change the order solutions come in but never the
set. Run the whole suite by running run_tests.py in the repo root."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from classes import Puzzle
from constants import UNPLACED
from serialization import load_game
from solving import solve_puzzle
from tests._helpers import ROOT, load, rows, write_pentomino_game

# None of these may change the solution set. branch="block" is left out on
# open boards, where it finds nothing in 30 s (see solver.solve_rows).
MODES = [
    {"branch": "cell"}, {"branch": "block"}, {"branch": "both"},
    {"order": True}, {"order": False}, {"seed": 0}, {"seed": 1, "order": True},
]
OPEN_MODES = [m for m in MODES if m.get("branch") != "block"]


class TestSolver(unittest.TestCase):
    def test_pro_booklet(self):
        """Every IQpuzzlerPRO puzzle's one solution is its booklet's
        (tests/data, checked by hand) -- except pyramid_puzzles/120, which
        has 5, contrary to the box."""
        game = load("IQpuzzlerPRO")
        for path in sorted((ROOT / "tests" / "data" / "IQpuzzlerPRO").glob("*.json")):
            book = game.books[path.stem]
            known = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(known), set(book))
            for name, (solution,) in known.items():
                with self.subTest(book=path.stem, puzzle=name):
                    puzzle = book[name]
                    # a list of layers of row strings: (depth, width) or (height, depth, width)
                    letters = np.array([list(row) for layer in solution for row in layer])
                    booklet = Puzzle(puzzle.setup, letters.reshape(puzzle.board.cells.shape[::-1]))
                    found = rows(puzzle)
                    self.assertIn(tuple(booklet.chosen_placement_idx.values()), found)
                    self.assertEqual(len(found), 5 if (path.stem, name) == ("pyramid_puzzles", "120") else 1)

    def test_pentominoes(self):
        """The 12 pentominoes tile 3x20 in 8 ways, in every mode."""
        with tempfile.TemporaryDirectory() as tmp:
            empty = load_game(write_pentomino_game(Path(tmp))).puzzles["empty_3x20"]
            expected = rows(empty, branch="cell")
            self.assertEqual((len(expected), len(set(expected))), (8, 8))
            for mode in OPEN_MODES:
                self.assertEqual(rows(empty, **mode), expected, mode)

    def test_modes_agree_on_valid_solutions(self):
        """Every mode finds the same solutions, and each is valid: its
        placements cover every cell exactly once and keep the puzzle's
        pre-placed blocks."""
        game = load("IQpuzzler")
        for book, name in (("main_puzzles", "50"), ("main_puzzles", "60"), ("pyramid_puzzles", "85")):
            puzzle = game.books[book][name]
            with self.subTest(book=book, puzzle=name):
                expected = rows(puzzle, branch="cell")
                self.assertEqual(len(expected), len(set(expected)), "duplicate solutions")
                for row in expected:
                    cells = np.concatenate([puzzle.placement_cells[i][p] for i, p in enumerate(row)])
                    self.assertEqual(sorted(cells.tolist()), list(range(puzzle.setup.n_cells)))
                    for i, p in puzzle.chosen_placement_idx.items():
                        if p != UNPLACED:
                            self.assertEqual(row[i], p)
                for mode in MODES:
                    self.assertEqual(rows(puzzle, **mode), expected, mode)

    def test_limits(self):
        """max_solutions is exact in every mode, a time limit stops an open
        board, and a run cut short either way is not `complete`."""
        game = load("IQpuzzler")
        empty, puzzle = game.puzzles["empty_main"], game.books["main_puzzles"]["50"]
        for mode in OPEN_MODES:
            for n in (0, 1, 7):
                found = rows(empty, max_solutions=n, **mode)
                self.assertEqual((len(found), len(set(found))), (n, n), mode)

        _, stats = solve_puzzle(empty, time_limit=0.2)
        self.assertFalse(stats.complete)
        self.assertLess(stats.duration, 1.0)
        _, stats = solve_puzzle(puzzle, max_solutions=3)
        self.assertEqual((len(stats.elapsed), stats.complete), (3, False))
        _, stats = solve_puzzle(puzzle)
        self.assertEqual((len(stats.elapsed), stats.complete), (101, True))
