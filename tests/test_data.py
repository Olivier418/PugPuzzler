"""Sanity checks on the puzzle data files (fast; no solving)."""
import unittest

from constants import EMPTY, UNPLACED
from tests._helpers import load

DIFFICULTIES = {"starter", "junior", "expert", "master", "wizard"}
GAMES = ["IQpuzzler", "IQpuzzlerPRO", "IQquub"]


class TestPuzzleData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = {n: load(n) for n in GAMES}

    def test_blocks_exactly_fill_board(self):
        for gname, game in self.games.items():
            for key, setup in game.setups.items():
                with self.subTest(game=gname, board=key):
                    self.assertEqual(sum(len(b.coords) for b in setup.blocks.values()), setup.n_cells)

    def test_book_puzzles(self):
        for gname, game in self.games.items():
            for bname, book in game.books.items():
                self.assertGreater(len(book), 0)
                for name, puzzle in book.items():
                    with self.subTest(game=gname, book=bname, puzzle=name):
                        self.assertIn(puzzle.difficulty, DIFFICULTIES)
                        self.assertEqual(puzzle.name, name)
                        placed = [i for i, p in puzzle.chosen_placement_idx.items() if p != UNPLACED]
                        self.assertLess(len(placed), len(puzzle.blocks), "puzzle is already solved")
                        self.assertGreater(len(placed), 0, "puzzle has no pre-filled blocks")
                        # given cells are exactly the union of the pre-placed blocks
                        n_given = int((puzzle.grid != EMPTY).sum())
                        self.assertEqual(n_given, sum(len(puzzle.blocks[i].coords) for i in placed))
