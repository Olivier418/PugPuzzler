"""Plotting smoke tests: the book plots take (puzzles, stats), draw without
raising, and read the solution count off the stats. Nothing is shown or
saved (Agg backend), and nothing is read from or written to solutions/."""
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- the backend must be chosen first

from classes import PuzzleBook  # noqa: E402
from plotting.plot_puzzle_stats import plot_puzzle_stats  # noqa: E402
from plotting.plot_solve_timeline import plot_solve_timeline  # noqa: E402
from solving import solve_puzzlebook  # noqa: E402
from tests._helpers import load  # noqa: E402


class TestBookPlots(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        source = load("IQpuzzler").books["main_puzzles"]
        cls.book = PuzzleBook(source["50"], source["51"], name="two")
        cls.solutions, cls.stats = solve_puzzlebook(cls.book)

    def tearDown(self):
        plt.close("all")

    def test_plot_puzzle_stats_2d_and_3d(self):
        ax = plot_puzzle_stats(self.book, self.stats)
        plotted = {y for _, y in ax.collections[0].get_offsets()}
        self.assertEqual(plotted, {len(sol.grids) for sol in self.solutions.values()})
        plot_puzzle_stats(self.book, self.stats, z="solve_time")

    def test_plot_solve_timeline(self):
        ax = plot_solve_timeline(self.book, self.stats)
        self.assertEqual([t.get_text() for t in ax.get_xticklabels()], ["50", "51"])

    def test_puzzles_must_cover_the_stats(self):
        only_50 = PuzzleBook(self.book["50"], name="one")
        with self.assertRaises(ValueError):
            plot_puzzle_stats(only_50, self.stats)
        with self.assertRaises(ValueError):
            plot_solve_timeline(only_50, self.stats)


if __name__ == "__main__":
    unittest.main()
