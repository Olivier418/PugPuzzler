"""Plotting smoke test: every book plot draws a solved book (Agg backend,
nothing shown or saved). plot_bounds is exercised in test_bounds."""
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- the backend must be chosen first

from benchmark import config_key  # noqa: E402
from classes import PuzzleBook  # noqa: E402
from plotting.plot_benchmark import plot_benchmark  # noqa: E402
from plotting.plot_puzzle_stats import plot_puzzle_stats  # noqa: E402
from plotting.plot_solve_timeline import plot_solve_timeline  # noqa: E402
from solving import solve_puzzlebook  # noqa: E402
from tests._helpers import load  # noqa: E402


class TestPlots(unittest.TestCase):
    def test_plots_draw(self):
        main = load("IQpuzzler").books["main_puzzles"]
        book = PuzzleBook(main["50"], main["60"], name="two")
        _, stats = solve_puzzlebook(book)
        ax = plot_puzzle_stats(book, stats)
        self.assertEqual({y for _, y in ax.collections[0].get_offsets()}, {101, 17})  # the solution counts
        plot_puzzle_stats(book, stats, z="solve_time")
        plot_solve_timeline(book, stats)
        plot_benchmark({config_key({}): list(stats.values())})
        plt.close("all")
