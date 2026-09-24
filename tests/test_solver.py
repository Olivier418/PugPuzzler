"""Solver tests. Run the whole suite by running run_tests.py in the repo root.

No test reads or writes solutions/stats on disk; known solutions are
written out literally in tests/data/ so they can be checked by eye."""
import time
import unittest

import numpy as np

from classes import Puzzle, Solution, Solver
from constants import EMPTY, UNPLACED
from solving import solve_puzzle
from tests._helpers import assert_valid_solution, load, load_known_solutions, unsolvable_puzzle


# PRO puzzles whose solution count contradicts the distributor's claim of one;
# see test_pyramid_120_has_one_solution.
KNOWN_MULTIPLE = {("pyramid_puzzles", "120")}


def known_puzzles(puzzle, solutions) -> list[Puzzle]:
    """Puzzles filled in from literal letter grids, one per solution.
    Constructing a Puzzle validates that every piece is a legal placement."""
    shape = puzzle.board.cells.shape[::-1]  # (depth, width) as in the puzzle JSON
    arrs = [np.array([list(row) for layer in solution for row in layer]).reshape(shape) for solution in solutions]
    return [Puzzle(puzzle.setup, arr, name=puzzle.name) for arr in arrs]


def known_solutions(puzzle, solutions) -> list[Solution]:
    """The literal solutions as Solution objects built in code."""
    return [Solution(puzzle, [puzzle.setup.to_full_grid(k.grid)]) for k in known_puzzles(puzzle, solutions)]


def solver_grids(puzzle) -> set:
    """Every solution the solver finds, as full board-shaped grids' bytes."""
    return {puzzle.setup.to_full_grid(s.grid).tobytes() for s in puzzle.solve()}


class TestKnownSolutions(unittest.TestCase):
    """A human-checkable solution must be among those the solver finds."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")
        cls.known = load_known_solutions("IQpuzzler")

    def test_known_solution_is_valid_and_found(self):
        for (book, name), solutions in self.known.items():
            with self.subTest(book=book, puzzle=name):
                puzzle = self.game.books[book][name]
                for known in known_puzzles(puzzle, solutions):
                    assert_valid_solution(self, puzzle, known)
                found = solver_grids(puzzle)
                for known in known_solutions(puzzle, solutions):
                    self.assertIn(known.grids[0].tobytes(), found)


class TestPro(unittest.TestCase):
    """IQpuzzlerPRO: the booklet's solutions are found, and every puzzle
    has exactly one solution (the distributor's claim)."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzlerPRO")
        cls.known = load_known_solutions("IQpuzzlerPRO")
        cls.found = {
            (book_name, name): solver_grids(puzzle)
            for book_name, book in cls.game.books.items()
            for name, puzzle in book.items()
        }

    def test_every_puzzle_has_a_booklet_solution(self):
        self.assertEqual(set(self.known), set(self.found))

    def test_booklet_solutions_are_valid_and_found(self):
        for (book, name), solutions in self.known.items():
            with self.subTest(book=book, puzzle=name):
                puzzle = self.game.books[book][name]
                for known in known_puzzles(puzzle, solutions):
                    assert_valid_solution(self, puzzle, known)
                for known in known_solutions(puzzle, solutions):
                    self.assertIn(known.grids[0].tobytes(), self.found[(book, name)])

    def test_exactly_one_solution(self):
        for (book, name), found in self.found.items():
            if (book, name) in KNOWN_MULTIPLE:
                continue
            with self.subTest(book=book, puzzle=name):
                self.assertEqual(len(found), 1, f"{len(found)} solutions")

    @unittest.expectedFailure
    def test_pyramid_120_has_one_solution(self):
        """The distributor's claim does not hold here: the puzzle has 5."""
        self.assertEqual(len(self.found[("pyramid_puzzles", "120")]), 1)


class TestUnsolvable(unittest.TestCase):
    def test_two_pieces_that_fit_alone_but_not_together(self):
        puzzle = unsolvable_puzzle(load("IQpuzzler"))

        empty = np.flatnonzero(puzzle.grid == EMPTY)
        unplaced = [i for i, p in puzzle.chosen_placement_idx.items() if p == UNPLACED]
        for i in unplaced:
            fits = [np.isin(pl, empty).all() for pl in puzzle.placement_cells[i]]
            self.assertTrue(any(fits), f"{puzzle.blocks[i].letter} should fit on its own")

        self.assertEqual(list(puzzle.solve()), [])


class TestSeeds(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")

    def test_seeds_give_same_solution_set(self):
        for book, name in [("main_puzzles", "40"), ("pyramid_puzzles", "85")]:
            with self.subTest(puzzle=name):
                puzzle = self.game.books[book][name]
                sets = [{s.grid.tobytes() for s in puzzle.solve(seed=seed)} for seed in (None, 0, 1)]
                self.assertEqual(sets[0], sets[1])
                self.assertEqual(sets[0], sets[2])
                for s in puzzle.solve():
                    assert_valid_solution(self, puzzle, s)


class TestLimits(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")
        cls.empty = cls.game.puzzles["empty_main"]
        cls.puzzle = cls.game.books["main_puzzles"]["40"]
        cls.all_grids = solver_grids(cls.puzzle)

    def test_max_solutions_edge_cases(self):
        n = len(self.all_grids)
        self.assertEqual(list(self.puzzle.solve(max_solutions=0)), [])
        one = list(self.puzzle.solve(max_solutions=1))
        self.assertEqual(len(one), 1)
        self.assertIn(self.puzzle.setup.to_full_grid(one[0].grid).tobytes(), self.all_grids)
        self.assertEqual(len(list(self.puzzle.solve(max_solutions=n + 10))), n)

    def test_time_limit(self):
        start = time.perf_counter()
        list(Solver(self.empty).solve(time_limit=0.5))
        self.assertLess(time.perf_counter() - start, 5.0)
        self.assertEqual(list(self.puzzle.solve(time_limit=0)), [])

    def test_propagates_through_puzzle_and_solution(self):
        self.assertEqual(len(list(self.empty.solve(seed=0, max_solutions=2))), 2)
        solution, _ = solve_puzzle(self.empty, seed=0, save=False, max_solutions=2)
        self.assertEqual(len(solution.grids), 2)
