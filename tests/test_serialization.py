"""Serialization tests: letter-grid JSON helpers and the solutions.json round
trip. Writes only into a temporary folder, never solutions/."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from classes import Solution, SolutionBook, SolveStats, SolveStatsBook
from serialization import (
    dump_json, letter_grid_to_rows, parse_letter_grid, save_solution_run, load_solution_run,
)
from tests._helpers import load


class TestLetterGrid(unittest.TestCase):
    def test_round_trip_flat_and_layered(self):
        for rows in (["AB ", "CDE"], [["AB", "CD"], ["E ", "  "]]):
            arr = parse_letter_grid(rows)
            self.assertEqual(letter_grid_to_rows(arr.T), rows)

    def test_dump_puts_one_row_per_line_aligned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            dump_json({"k": [["AB", "CD"]], "n": [1, 2]}, path)
            text = path.read_text()
            self.assertEqual(json.loads(text), {"k": [["AB", "CD"]], "n": [1, 2]})
            self.assertIn('  "k": [\n    ["AB",\n     "CD"]\n  ]', text)


class TestSolutionRoundTrip(unittest.TestCase):
    def test_solutions_saved_as_letters_and_reloaded(self):
        game = load("IQpuzzler")
        for book_name, name in (("main_puzzles", "50"), ("pyramid_puzzles", "85")):
            with self.subTest(book=book_name, puzzle=name):
                puzzle = game.books[book_name][name]
                solution, stats = Solution.from_puzzle(puzzle, save=False)
                self.assertTrue(solution.grids)
                with tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp) / "IQpuzzler" / "books" / book_name / name / "result_1"
                    save_solution_run(SolutionBook(solution), SolveStatsBook(stats), folder)
                    text = (folder / "solutions.json").read_text()
                    self.assertNotIn("0,", text)  # no block indices
                    solution_book, _ = load_solution_run(folder)
                    loaded = solution_book[name]
                    self.assertEqual(len(loaded.grids), len(solution.grids))
                    for a, b in zip(loaded.grids, solution.grids):
                        self.assertTrue(np.array_equal(a, b))


if __name__ == "__main__":
    unittest.main()
