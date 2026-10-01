"""Serialization tests: runs saved and loaded back, and a new book given a
home in (a copy of) its game. Writes only into temporary folders."""
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from classes import PuzzleBook
from serialization import load_complete_run, load_run_books, save_puzzlebook
from solving import solve_puzzle, solve_puzzlebook
from tests._helpers import GAMES_ROOT, load


class TestRuns(unittest.TestCase):
    def test_puzzle_run(self):
        """A pyramid puzzle's run (3D letter grids) loads back as the same
        rows and stats, on the game's own Setup; load_complete_run refuses a
        run cut short and hands back a complete one."""
        puzzle = load("IQpuzzler").books["pyramid_puzzles"]["85"]
        with tempfile.TemporaryDirectory() as tmp:
            save = dict(save_solutions=True, save_stats=True, solutions_root=tmp)
            solve_puzzle(puzzle, max_solutions=1, **save)
            solution, stats = solve_puzzle(puzzle, **save)
            runs = Path(tmp) / puzzle.source.relative_dir()

            puzzles, solutions, loaded_stats = load_run_books(runs / "result_2", games_root=GAMES_ROOT)
            self.assertEqual(list(puzzles), ["85"])
            self.assertIs(puzzles["85"].setup, puzzle.setup)
            self.assertTrue(np.array_equal(solutions["85"].rows, solution.rows))
            self.assertEqual(loaded_stats["85"], stats)

            with self.assertRaises(ValueError):
                load_complete_run(runs / "result_1", puzzle)
            complete, complete_stats = load_complete_run(runs / "result_2", puzzle)
            self.assertTrue(np.array_equal(complete.rows, solution.rows))
            self.assertEqual(complete_stats, stats)

    def test_new_book_run(self):
        """A book made in memory gets a home with save_puzzlebook, after
        which its run saves and loads like any other."""
        game = load("IQpuzzler")
        main = game.books["main_puzzles"]
        book = PuzzleBook(main["50"].copy(rename="1"), main["60"].copy(rename="2"), name="made_here")
        with tempfile.TemporaryDirectory() as tmp:
            games_root = Path(tmp) / "games"
            shutil.copytree(GAMES_ROOT / "IQpuzzler", games_root / "IQpuzzler")
            save_puzzlebook(book, game, games_root=games_root)
            solutions, stats = solve_puzzlebook(book, save_solutions=True, save_stats=True, solutions_root=tmp)

            folder = Path(tmp) / "IQpuzzler" / "books" / "made_here" / "result_1"
            puzzles, loaded_solutions, loaded_stats = load_run_books(folder, games_root=games_root)
            self.assertEqual(list(puzzles), ["1", "2"])
            self.assertEqual(loaded_stats, stats)
            for name, solution in solutions.items():
                self.assertTrue(np.array_equal(puzzles[name].grid, book[name].grid))
                self.assertTrue(np.array_equal(loaded_solutions[name].rows, solution.rows))
