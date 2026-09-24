"""Serialization tests: letter-grid JSON helpers and the run-folder round
trip. Writes only into a temporary folder, never solutions/."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from classes import Game, PuzzleBook
from constants import UNPLACED
from serialization import (
    dump_json, letter_grid_to_rows, load_puzzles, load_solution_run, parse_letter_grid,
    save_puzzlebook, save_solution_run,
)
from solving import single_run_books, solve_puzzle, solve_puzzlebook
from tests._helpers import load


class TestLetterGrid(unittest.TestCase):
    def test_round_trip_flat_and_layered(self):
        for rows in (["AB ", "CDE"], [["AB", "CD"], ["E ", "  "]]):
            arr = parse_letter_grid(rows)
            self.assertEqual(letter_grid_to_rows(arr.T), rows)

    def test_dump_json_round_trips(self):
        data = {"k": [["AB", "CD"]], "n": [1, 2]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            dump_json(data, path)
            self.assertEqual(json.loads(path.read_text()), data)


def assert_same_run(test, loaded, original):
    """Two (SolutionBook, SolveStatsBook) pairs hold the same puzzles, solutions and stats."""
    (loaded_sols, loaded_stats), (sols, stats) = loaded, original
    test.assertEqual(list(loaded_sols), list(sols))
    for name, sol in sols.items():
        got = loaded_sols[name]
        test.assertEqual(got.puzzle.difficulty, sol.puzzle.difficulty)
        test.assertTrue(np.array_equal(got.puzzle.grid, sol.puzzle.grid))
        test.assertEqual(len(got.grids), len(sol.grids))
        for a, b in zip(got.grids, sol.grids):
            test.assertTrue(np.array_equal(a, b))
        test.assertEqual(loaded_stats[name], stats[name])


def hand_made(puzzle, name):
    """`puzzle` with one pre-placed piece taken off, as a new puzzle that
    exists nowhere in games/."""
    made = puzzle.copy(rename=name)
    made.source = None
    made.remove(next(i for i, p in made.chosen_placement_idx.items() if p != UNPLACED))
    return made


class TestSolutionRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")

    def test_single_puzzle_round_trip(self):
        for book_name, name in (("main_puzzles", "50"), ("pyramid_puzzles", "85")):
            with self.subTest(book=book_name, puzzle=name):
                puzzle = self.game.books[book_name][name]
                run = single_run_books(*solve_puzzle(puzzle))
                self.assertTrue(run[0][name].grids)
                with tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp) / "IQpuzzler" / "books" / book_name / name / "result_1"
                    save_solution_run(*run, folder)
                    self.assertEqual(sorted(f.name for f in folder.iterdir()), ["solutions.json", "stats.json"])
                    self.assertRegex((folder / "solutions.json").read_text(), r'^[\[\]",\sA-Za-z]*$')
                    assert_same_run(self, load_solution_run(folder), run)  # puzzles from games/
                    loaded, _ = load_solution_run(folder, game=self.game)
                    self.assertIs(loaded[name].puzzle, puzzle)  # ... or the ones in memory

    def test_whole_book_round_trip(self):
        source = self.game.books["main_puzzles"]
        book = PuzzleBook(source["50"], source["51"], name="two")
        game = Game(books=[book], setups=self.game.setups, name="IQpuzzler")
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "IQpuzzler" / "books" / "two" / "result_1"
            run = solve_puzzlebook(book, save=True, path=folder)
            assert_same_run(self, load_solution_run(folder, game=game), run)

    def test_new_book_gets_a_home_before_its_run_is_saved(self):
        game = load("IQpuzzler")
        book = PuzzleBook(hand_made(game.books["main_puzzles"]["50"], "A"), name="made_here")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):  # nowhere in games/ yet: nothing to refer to
                solve_puzzlebook(book, save=True, solutions_root=tmp)

            json_file = save_puzzlebook(book, game, games_root=Path(tmp) / "games")
            self.assertIs(game.books["made_here"], book)
            reread = load_puzzles(json_file, game.setups)
            self.assertTrue(np.array_equal(reread[0].grid, book["A"].grid))
            with self.assertRaises(FileExistsError):
                save_puzzlebook(book, game, games_root=Path(tmp) / "games")

            run = solve_puzzlebook(book, save=True, solutions_root=tmp)
            folder = Path(tmp) / "IQpuzzler" / "books" / "made_here" / "result_1"
            assert_same_run(self, load_solution_run(folder, game=game), run)


if __name__ == "__main__":
    unittest.main()
