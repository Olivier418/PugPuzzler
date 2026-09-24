"""Branching tests: every `branch` / `order` mode must agree on the solution
set, and the search must agree with counts published outside this project.

Run the whole suite by running run_tests.py in the repo root. Nothing here
reads or writes solutions/ or benchmarks/.

`branch="cell"` is the baseline the rest are checked against -- it is the
narrowest of the three rules and derives nothing from anything else.

`branch="block"` is slow by design: it cannot see an uncoverable cell until
some block runs out of room, so with nothing pre-filled to constrain it there
is almost nothing to prune. On a book puzzle that costs a factor (worst here:
main_puzzles/65, 2 s against 0.01 s); on a wide-open board it is hopeless --
the empty main board yields no solution at all in 30 s. So it is exercised on
the constrained fixtures only, and OPEN_MODES leaves it out of the two tests
that start from an open board.

The synthetic cases use the 12 pentominoes on a rectangle, whose solution
count is published and was not derived from this solver -- so they check the
search against an outside number."""
import unittest

import numpy as np

from classes import Block, BlockCollection, Puzzle, RegularBoard, Setup, Solver, State
from tests._helpers import assert_valid_solution, load, unsolvable_puzzle


# The three branch rules, plus candidate ordering forced on and off. None of
# them may change the solution set.
MODES = [
    {"branch": "cell"}, {"branch": "block"}, {"branch": "both"},
    {"order": True}, {"order": False},
]

# The same, minus "block", for cases that start from an open board -- see the
# module docstring.
OPEN_MODES = [m for m in MODES if m != {"branch": "block"}]

PENTOMINOES = {
    "F": [".XX", "XX.", ".X."], "I": ["XXXXX"],        "L": ["X.", "X.", "X.", "XX"],
    "N": [".X", ".X", "XX", "X."], "P": ["XX", "XX", "X."], "T": ["XXX", ".X.", ".X."],
    "U": ["X.X", "XXX"],        "V": ["X..", "X..", "XXX"], "W": ["X..", "XX.", ".XX"],
    "X": [".X.", "XXX", ".X."], "Y": [".X", "XX", ".X", ".X"], "Z": ["XX.", ".X.", ".XX"],
}


def pentomino_setup(width: int, depth: int) -> Setup:
    blocks = [
        Block(np.array([[c == "X" for c in row] for row in rows], dtype=bool),
              (128, 128, 128), letter, "white")
        for letter, rows in PENTOMINOES.items()
    ]
    return Setup(BlockCollection(*blocks), RegularBoard(width=width, depth=depth))


def grids(state, **kwargs) -> list:
    """Every solution the solver finds, as grid bytes, sorted."""
    return sorted(s.grid.tobytes() for s in Solver(state).solve(**kwargs))


class TestModesAgree(unittest.TestCase):
    """branch and order change the route, never the destination."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")

    def cases(self):
        for name in ("11", "12", "17", "20", "22", "35", "65", "45"):
            yield "main_puzzles", name
        for name in ("84", "85", "88"):
            yield "pyramid_puzzles", name

    def test_same_solution_set(self):
        for book, name in self.cases():
            puzzle = self.game.books[book][name]
            with self.subTest(book=book, puzzle=name):
                expected = grids(puzzle, branch="cell")
                self.assertEqual(len(expected), len(set(expected)), "duplicate solutions")
                for mode in MODES:
                    self.assertEqual(grids(puzzle, **mode), expected, f"{mode} disagrees")

    def test_solutions_are_valid(self):
        """A solution's grid and its chosen_placement_idx are written
        together by place_unchecked, but nothing else guarantees they
        agree with the puzzle they came from."""
        for book, name in (("main_puzzles", "17"), ("main_puzzles", "11"),
                           ("pyramid_puzzles", "84")):
            puzzle = self.game.books[book][name]
            with self.subTest(book=book, puzzle=name):
                for mode in MODES:
                    for solution in Solver(puzzle).solve(**mode):
                        assert_valid_solution(self, puzzle, solution)

    def test_unsolvable_stays_unsolvable(self):
        """The same fixture as test_solver.TestUnsolvable, but under every
        mode. F and H each fit the leftover hole alone, never together --
        so a mode that mistakenly prunes or derives will show up here."""
        puzzle = unsolvable_puzzle(self.game)
        for mode in MODES:
            with self.subTest(**mode):
                self.assertEqual(grids(puzzle, **mode), [])


class TestPentominoes(unittest.TestCase):
    """A published count, independent of this solver: the 12 pentominoes tile
    3x20 in 8 ways.

    Only 3x20 runs here -- it is the one that fits the suite's time budget.
    4x15 (1472), 5x12 (4040) and 6x10 (9356) are the same check at minutes
    rather than seconds, worth running by hand after touching the kernel."""

    CASES = [(20, 3, 8)]

    def test_counts(self):
        for width, depth, total in self.CASES:
            setup = pentomino_setup(width, depth)
            state = State(setup)
            with self.subTest(board=f"{depth}x{width}"):
                baseline = grids(state, branch="cell")
                self.assertEqual(len(baseline), total)
                for mode in OPEN_MODES:
                    self.assertEqual(grids(state, **mode), baseline, f"{mode} disagrees")


class TestLimits(unittest.TestCase):
    """max_solutions is exact in every mode, and an early stop must still
    leave the solver usable."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")
        cls.empty = cls.game.puzzles["empty_main"]

    def test_max_solutions_is_exact(self):
        for mode in OPEN_MODES:
            with self.subTest(**mode):
                for n in (0, 1, 2, 3, 5, 7):
                    found = list(Solver(self.empty).solve(seed=0, max_solutions=n, **mode))
                    self.assertEqual(len(found), n)
                    self.assertEqual(len({s.grid.tobytes() for s in found}), n, "duplicates")

    def test_solutions_are_valid_and_distinct(self):
        for mode in OPEN_MODES:
            with self.subTest(**mode):
                found = list(Solver(self.empty).solve(seed=0, max_solutions=25, **mode))
                self.assertEqual(len({s.grid.tobytes() for s in found}), 25)
                for state in found:
                    assert_valid_solution(self, self.empty, state)


class TestBadOptions(unittest.TestCase):
    """branch and order reject anything they don't know, rather than
    silently falling back to a default."""

    @classmethod
    def setUpClass(cls):
        cls.state = load("IQpuzzler").books["main_puzzles"]["50"]

    def test_unknown_names_raise(self):
        for kwargs in ({"branch": "mrv"}, {"branch": "first"}, {"branch": 3},
                       {"order": "counts_"}, {"order": 2}):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    list(Solver(self.state).solve(max_solutions=1, **kwargs))


if __name__ == "__main__":
    unittest.main()
