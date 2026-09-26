"""Solution-bound tests: the subset grouping against brute force, the whole
bounds on a small board, and their plot. Run the whole suite by running
run_tests.py in the repo root. Everything is written to a temporary folder
only, never games/, solutions/ or bounds/; plots use the Agg backend and are
never shown or saved."""
import itertools
import json
import tempfile
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- the backend must be chosen first
import numpy as np  # noqa: E402

from pathlib import Path  # noqa: E402

from plotting.plot_bounds import plot_bounds  # noqa: E402
from plotting.plot_puzzle_stats import plot_puzzle_stats  # noqa: E402
from puzzle_bounds import bound_counts, compute_puzzle_bounds  # noqa: E402
from constants import LOWER_BOUND_BOOK, UPPER_BOUND_BOOK  # noqa: E402
from serialization import load_game  # noqa: E402
from solving import empty_puzzle, solve_puzzle  # noqa: E402
from tests.test_branching import PENTOMINOES  # noqa: E402


def write_pentomino_game(games_root: Path) -> Path:
    """A game folder "Pentominoes" under `games_root`: the 12 pentominoes on
    one 3x20 board (8 solutions), with its empty puzzle -- small enough for
    the bounds to be computed from scratch in the suite."""
    folder = games_root / "Pentominoes"
    (folder / "puzzles").mkdir(parents=True)
    blocks = {
        letter: {
            "letter": letter, "rgb": [128, 128, 128],
            "positions": [[int(c == "X") for c in row] for row in rows],
            "terminal": {"bg": "WHITE", "fg": "BLACK"},
        }
        for letter, rows in PENTOMINOES.items()
    }
    (folder / "blocks.json").write_text(json.dumps(blocks))
    (folder / "boards.json").write_text(json.dumps({"3x20": {"type": "RegularBoard", "width": 20, "depth": 3}}))
    (folder / "puzzles" / "empty_3x20.json").write_text(json.dumps({"board": "3x20"}))
    return folder


def brute_force(table, sizes):
    """{filled: (fewest, most)} over every subset of columns, by grouping the
    rows on all of the subset's columns at once."""
    n = table.shape[1]
    out = {0: (len(table), len(table))}
    for k in range(1, n):
        for cols in itertools.combinations(range(n), k):
            _, counts = np.unique(table[:, cols], axis=0, return_counts=True)
            f = sum(sizes[c] for c in cols)
            lo, hi = out.get(f, (np.inf, 0))
            out[f] = (min(lo, int(counts.min())), max(hi, int(counts.max())))
    return out


def count_matching(table, realization):
    """How many rows place every column of `realization` its way."""
    mask = np.ones(len(table), dtype=bool)
    for col, placement_idx in realization.items():
        mask &= table[:, col] == placement_idx
    return int(mask.sum())


class TestBoundCounts(unittest.TestCase):
    """The DFS (with its settling and row dropping) finds exactly what
    grouping every subset from scratch finds, and every realization it
    reports really has the count it claims."""

    def test_against_brute_force(self):
        rng = np.random.default_rng(0)
        for trial in range(3):
            with self.subTest(trial=trial):
                n_placements = rng.integers(3, 7, size=6).tolist()
                sizes = rng.integers(1, 4, size=6).tolist()
                table = np.stack([rng.integers(0, p, size=2000) for p in n_placements], axis=1)
                lower, upper = bound_counts(table, n_placements, sizes)

                expected = brute_force(table, sizes)
                self.assertEqual(set(lower), set(expected))
                self.assertEqual(set(upper), set(expected))
                for f, (lo, hi) in expected.items():
                    self.assertEqual(lower[f][0], lo, f"lower bound at {f}")
                    self.assertEqual(upper[f][0], hi, f"upper bound at {f}")
                    for count, realization in (lower[f], upper[f]):
                        self.assertEqual(sum(sizes[c] for c in realization), f)
                        self.assertEqual(count_matching(table, realization), count)


class TestBounds(unittest.TestCase):
    """The whole bounds on the 12 pentominoes on 3x20 (8 solutions): every
    puzzle's stats agree with a fresh solve, lower <= upper everywhere, the
    empty board's solve must be complete, and the bounds plot."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.game = load_game(write_pentomino_game(Path(cls.tmp.name) / "games"))
        cls.empty = solve_puzzle(empty_puzzle(cls.game, "3x20"))
        cls.bounds = compute_puzzle_bounds(*cls.empty, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def tearDown(self):
        plt.close("all")

    def test_books(self):
        (lower_puzzles, _), (upper_puzzles, _) = self.bounds
        self.assertEqual(lower_puzzles.name, LOWER_BOUND_BOOK)
        self.assertEqual(upper_puzzles.name, UPPER_BOUND_BOOK)
        self.assertEqual({p.difficulty for p in lower_puzzles.values()}, {"hardest"})
        self.assertEqual({p.difficulty for p in upper_puzzles.values()}, {"easiest"})
        _, empty_stats = self.empty
        for puzzles, stats in self.bounds:
            self.assertEqual(list(puzzles), list(stats))
            self.assertEqual(stats["1"], empty_stats._replace(puzzle_name="1"))  # the empty board
            for name, puzzle in puzzles.items():
                with self.subTest(book=puzzles.name, puzzle=name):
                    self.assertEqual(len(stats[name].elapsed), len(list(puzzle.solve())))

    def counts(self):
        """({filled: lower bound}, {filled: upper bound})."""
        return tuple(
            {p.nr_filled_cells: len(stats[n].elapsed) for n, p in puzzles.items()} for puzzles, stats in self.bounds
        )

    def test_lower_is_at_most_upper(self):
        lows, highs = self.counts()
        self.assertEqual(set(lows), set(highs))
        for filled in lows:
            self.assertLessEqual(lows[filled], highs[filled])
            self.assertGreaterEqual(lows[filled], 1)

    def test_needs_a_complete_empty_solve(self):
        cut_short = solve_puzzle(empty_puzzle(self.game, "3x20"), max_solutions=3)
        with self.assertRaises(ValueError):
            compute_puzzle_bounds(*cut_short, verbose=False)

    def test_plot_bounds(self):
        (lower_puzzles, lower_stats), (upper_puzzles, upper_stats) = self.bounds
        ax = plot_bounds(lower_puzzles, lower_stats, upper_puzzles, upper_stats)
        lows, highs = self.counts()
        dashed = [line for line in ax.lines if line.get_linestyle() == "--"]
        self.assertEqual({line.get_marker() for line in dashed}, {"o"})
        self.assertEqual(
            [list(zip(*line.get_data())) for line in dashed],
            [sorted(lows.items()), sorted(highs.items())],
        )
        self.assertEqual(len(ax.collections), 2)  # the two shadings; the dots are the dashed lines' markers
        self.assertEqual(len(ax.texts), 0)  # and no names
        self.assertEqual(ax.get_yscale(), "log")

        # an overlay inherits the axes and can't rescale the plot
        limits = ax.get_xlim(), ax.get_ylim()
        plot_puzzle_stats({"2": lower_puzzles["2"]}, {"2": lower_stats["2"]}, ax=ax)
        self.assertEqual((ax.get_xlim(), ax.get_ylim()), limits)
        self.assertEqual(len(ax.collections), 3)
        self.assertEqual(ax.get_title(), "Solution bounds")
