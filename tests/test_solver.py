"""Solver tests. Run the whole suite by running run_tests.py in the repo root.

No test reads or writes solutions/stats on disk; known solutions are
written out literally in tests/data/ so they can be checked by eye."""
import itertools
import unittest

import numpy as np

from classes import Puzzle, Solution
from Solver import Solver
from constants import EMPTY, UNPLACED
from tests._helpers import assert_valid_solution, load, load_known_solutions


def known_puzzles(puzzle, solutions) -> list[Puzzle]:
    """Puzzles filled in from literal letter grids, one per solution.
    Constructing a Puzzle validates that every piece is a legal placement."""
    shape = puzzle.board.cells.shape[::-1]  # (depth, width) as in the puzzle JSON
    arrs = [np.array([list(row) for layer in solution for row in layer]).reshape(shape) for solution in solutions]
    return [Puzzle(puzzle.setup, arr, name=puzzle.name) for arr in arrs]


def known_solutions(puzzle, solutions) -> list[Solution]:
    """The literal solutions as Solution objects built in code."""
    return [Solution(puzzle.name, [puzzle.setup.expand(k.grid)], puzzle=puzzle) for k in known_puzzles(puzzle, solutions)]


def solver_grids(puzzle) -> set:
    """Every solution the solver finds, as full board-shaped grids' bytes."""
    return {puzzle.setup.expand(s.grid).tobytes() for s in puzzle.solve()}


class TestKnownSolutions(unittest.TestCase):
    """A human-checkable solution must be among those the solver finds."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")
        cls.known = load_known_solutions("IQpuzzler")

    def test_known_solution_is_found(self):
        for (book, name), solutions in self.known.items():
            with self.subTest(book=book, puzzle=name):
                puzzle = self.game.books[book][name]
                found = solver_grids(puzzle)
                for known in known_solutions(puzzle, solutions):
                    self.assertIn(known.grids[0].tobytes(), found)

    def test_known_solutions_are_valid(self):
        for (book, name), solutions in self.known.items():
            with self.subTest(book=book, puzzle=name):
                puzzle = self.game.books[book][name]
                for known in known_puzzles(puzzle, solutions):
                    assert_valid_solution(self, puzzle, known)


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
            with self.subTest(book=book, puzzle=name):
                self.assertEqual(len(found), 1, f"{len(found)} solutions")


class TestUnsolvable(unittest.TestCase):
    def test_two_pieces_that_fit_alone_but_not_together(self):
        game = load("IQpuzzler")
        setup = game.books["main_puzzles"]["1"].setup
        # Every piece is placed except F (3 cells) and H (5 cells); the
        # 8 empty cells (blank below) form a 3x3 block whose top-middle cell
        # is taken by D, i.e. a "U" opening upwards.
        # F fits inside it, and so does H, but never both at once.
        letters = [
            "E","E","G","G","G","J","J","J","J","I","I",
            "A","E","E","E","G","C","D","D","D","D","I",
            "A","A","A","L","G","C"," ","D"," ","I","I",
            "B","B","L","L","L","C"," "," "," ","K","K",
            "B","B","B","L","C","C"," "," "," ","K","K",
        ]
        puzzle = Puzzle(setup, np.array(letters).reshape(5, 11), name="unsolvable")

        empty = np.flatnonzero(puzzle.grid == EMPTY)
        unplaced = [i for i, p in puzzle.chosen_placement_idx.items() if p == UNPLACED]
        self.assertEqual(sorted(puzzle.blocks[i].letter for i in unplaced), ["F", "H"])
        self.assertEqual(len(empty), sum(len(puzzle.blocks[i].coords) for i in unplaced))
        for i in unplaced:
            fits = [np.isin(pl, empty).all() for pl in puzzle.placements[i]]
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


class TestEmptyBoards(unittest.TestCase):
    def test_first_solutions_valid_and_distinct(self):
        game = load("IQpuzzler")
        for name in ("empty_main", "empty_pyramid"):
            with self.subTest(puzzle=name):
                puzzle = game.puzzles[name]
                sols = list(itertools.islice(Solver(puzzle).solve(seed=0), 25))
                self.assertEqual(len(sols), 25)
                self.assertEqual(len({s.grid.tobytes() for s in sols}), 25)
                for s in sols:
                    assert_valid_solution(self, puzzle, s)
