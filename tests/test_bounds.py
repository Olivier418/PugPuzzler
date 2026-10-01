"""Solution-bound tests: the subset grouping against brute force, and the
whole pipeline on a small board. Plots use the Agg backend."""
import itertools
import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 -- the backend must be chosen first
import numpy as np  # noqa: E402

from plotting.plot_bounds import plot_bounds  # noqa: E402
from plotting.plot_puzzle_stats import plot_puzzle_stats  # noqa: E402
from puzzle_bounds import bound_counts, compute_puzzle_bounds  # noqa: E402
from serialization import load_bounds, load_game, save_bounds  # noqa: E402
from solving import solve_puzzle  # noqa: E402
from tests._helpers import write_pentomino_game  # noqa: E402


class TestBounds(unittest.TestCase):
    def test_bound_counts_against_brute_force(self):
        """The DFS (with its settling and row dropping) finds the same fewest
        and most solutions per filled count as grouping every subset from
        scratch, and every realization it reports has the count it claims."""
        rng = np.random.default_rng(0)
        for trial in range(3):
            n_placements = rng.integers(3, 7, size=6).tolist()
            sizes = rng.integers(1, 4, size=6).tolist()
            table = np.stack([rng.integers(0, p, size=2000) for p in n_placements], axis=1)
            lower, upper = bound_counts(table, n_placements, sizes)

            expected = {0: (len(table), len(table))}
            for k in range(1, table.shape[1]):
                for cols in itertools.combinations(range(table.shape[1]), k):
                    _, counts = np.unique(table[:, cols], axis=0, return_counts=True)
                    f = sum(sizes[c] for c in cols)
                    lo, hi = expected.get(f, (np.inf, 0))
                    expected[f] = (int(min(lo, counts.min())), int(max(hi, counts.max())))

            with self.subTest(trial=trial):
                self.assertEqual((lower.keys(), upper.keys()), (expected.keys(), expected.keys()))
                self.assertEqual({f: (lower[f][0], upper[f][0]) for f in expected}, expected)
                for f, (count, realization) in [*lower.items(), *upper.items()]:
                    self.assertEqual(sum(sizes[c] for c in realization), f)
                    matching = np.ones(len(table), dtype=bool)
                    for c, p in realization.items():
                        matching &= table[:, c] == p
                    self.assertEqual(matching.sum(), count)

    def test_pentomino_bounds(self):
        """The whole pipeline on the 12 pentominoes on 3x20 (8 solutions):
        compute_puzzle_bounds (which re-solves every bound puzzle and checks
        its count against the grouping's), save_bounds / load_bounds, and
        plot_bounds with a book overlaid."""
        with tempfile.TemporaryDirectory() as tmp:
            game = load_game(write_pentomino_game(Path(tmp) / "games"))
            bounds = compute_puzzle_bounds(*solve_puzzle(game.puzzles["empty_3x20"]), verbose=False)
            save_bounds(game, *bounds, root=tmp)
            loaded = load_bounds(game, "3x20", root=tmp)

        for (puzzles, stats), (got_puzzles, got_stats) in zip(bounds, loaded):
            self.assertEqual(got_stats, stats)
            self.assertEqual(list(got_puzzles), list(puzzles))
            for name, puzzle in puzzles.items():
                self.assertEqual(got_puzzles[name].difficulty, puzzle.difficulty)
                self.assertTrue(np.array_equal(got_puzzles[name].grid, puzzle.grid))

        (lower_puzzles, lower_stats), (upper_puzzles, upper_stats) = loaded
        ax = plot_bounds(lower_puzzles, lower_stats, upper_puzzles, upper_stats)
        plot_puzzle_stats(lower_puzzles, lower_stats, ax=ax)
        plt.close("all")
