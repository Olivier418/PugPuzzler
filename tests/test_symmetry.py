"""Symmetry tests: the geometry (Lattice.point_group, Setup.region_symmetries)
and the solver's orbit reduction on top of it.

Run the whole suite by running run_tests.py in the repo root. Nothing here
reads or writes solutions/ or benchmarks/.

The synthetic cases use the 12 pentominoes on a rectangle, whose solution
counts are published and were not derived from this solver -- so they check
both the plain search and the symmetry reduction against an outside number."""
import itertools
import unittest

import numpy as np

from classes import Block, BlockCollection, Puzzle, RegularBoard, Setup, Solver, State
from constants import EMPTY
from tests._helpers import assert_valid_solution, load


# Every (branching, symmetry) pair, plus candidate ordering forced on and
# off. None of them may change the solution set.
MODES = [
    {"branching": branching, "symmetry": symmetry}
    for branching in ("item", "cell", "block", "hybrid")
    for symmetry in (False, True)
] + [{"order": True}, {"order": False}]

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


def apply_image(grid, image):
    out = np.empty_like(grid)
    out[image] = grid
    return out


class TestGeometry(unittest.TestCase):
    """Setup.region_symmetries must return an actual group of permutations."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")

    def test_board_symmetry_counts(self):
        # An 11x5 rectangle: identity, half turn, two mirrors. The 5x5
        # pyramid: the square's dihedral group.
        self.assertEqual(len(self.game.setups["main"].board_symmetries), 4)
        self.assertEqual(len(self.game.setups["pyramid"].board_symmetries), 8)

    def test_point_group_sizes(self):
        self.assertEqual(len(self.game.setups["main"].board.lattice.point_group), 8)
        self.assertEqual(len(self.game.setups["pyramid"].board.lattice.point_group), 48)

    def test_group_axioms(self):
        for key, setup in self.game.setups.items():
            images = setup.board_symmetries
            keys = {im.tobytes() for im in images}
            with self.subTest(board=key):
                identity = np.arange(setup.n_cells)
                self.assertIn(identity.tobytes(), keys, "identity missing")
                for a, b in itertools.product(images, repeat=2):
                    # closed under composition, so inverses are in there too
                    self.assertIn(a[b].tobytes(), keys, "not closed")
                for im in images:
                    self.assertCountEqual(im.tolist(), identity.tolist(), "not a permutation")

    def test_identity_outside_the_region(self):
        """The whole point: an image only ever moves the region's own cells,
        so it can be applied to a full grid without disturbing anything
        already decided."""
        setup = self.game.setups["main"]
        puzzle = self.game.books["main_puzzles"]["17"]
        region = puzzle.grid == EMPTY
        for image in setup.region_symmetries(region):
            outside = np.flatnonzero(~region)
            self.assertTrue((image[outside] == outside).all())


class TestModesAgree(unittest.TestCase):
    """branching and symmetry change the route, never the destination."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")

    def cases(self):
        # The symmetric puzzles (|G| > 1 at the root) plus an ordinary one,
        # on both boards. Kept to puzzles that are quick under *every*
        # mode -- "block" on its own is slow by design.
        for name in ("11", "12", "17", "20", "22", "35", "65", "45"):
            yield "main_puzzles", name
        for name in ("84", "85", "88"):
            yield "pyramid_puzzles", name

    def test_same_solution_set(self):
        for book, name in self.cases():
            puzzle = self.game.books[book][name]
            with self.subTest(book=book, puzzle=name):
                expected = grids(puzzle, branching="cell", symmetry=False)
                self.assertEqual(len(expected), len(set(expected)), "duplicate solutions")
                for mode in MODES:
                    self.assertEqual(grids(puzzle, **mode), expected, f"{mode} disagrees")

    def test_unsolvable_stays_unsolvable(self):
        """The same fixture as test_solver.TestUnsolvable, but under every
        mode: the leftover hole is a symmetric "U", so this is exactly the
        shape that would tempt the orbit code into deriving a solution
        that does not exist. F and H each fit it alone, never together."""
        setup = self.game.books["main_puzzles"]["1"].setup
        letters = [
            "E", "E", "G", "G", "G", "J", "J", "J", "J", "I", "I",
            "A", "E", "E", "E", "G", "C", "D", "D", "D", "D", "I",
            "A", "A", "A", "L", "G", "C", " ", "D", " ", "I", "I",
            "B", "B", "L", "L", "L", "C", " ", " ", " ", "K", "K",
            "B", "B", "B", "L", "C", "C", " ", " ", " ", "K", "K",
        ]
        puzzle = Puzzle(setup, np.array(letters).reshape(5, 11), name="unsolvable")
        for mode in MODES:
            with self.subTest(**mode):
                self.assertEqual(grids(puzzle, **mode), [])
        self.assertEqual(grids(puzzle, up_to_symmetry=True), [])

    def test_derived_solutions_are_valid(self):
        """A derived solution is built by transforming a grid, so its
        chosen_placement_idx can silently disagree with it."""
        for book, name in (("main_puzzles", "17"), ("main_puzzles", "11"),
                           ("pyramid_puzzles", "84")):
            puzzle = self.game.books[book][name]
            with self.subTest(book=book, puzzle=name):
                for solution in puzzle.solve():
                    assert_valid_solution(self, puzzle, solution)


class TestUpToSymmetry(unittest.TestCase):
    """One solution per symmetry class, exactly."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")

    def test_known_counts(self):
        # IQpuzzler claims a unique solution per puzzle; where the solver
        # finds several, they are rotations/mirrors of one another.
        expected = {"11": 1, "12": 1, "17": 1, "18": 1, "19": 1, "20": 1,
                    "22": 2, "35": 1, "65": 1}
        for name, count in expected.items():
            puzzle = self.game.books["main_puzzles"][name]
            with self.subTest(puzzle=name):
                self.assertEqual(len(grids(puzzle, up_to_symmetry=True)), count)

    def test_matches_full_count_without_symmetry(self):
        """On a puzzle whose open region is asymmetric there is nothing to
        collapse, so it must agree with an ordinary solve."""
        for name in ("45", "50"):
            puzzle = self.game.books["main_puzzles"][name]
            with self.subTest(puzzle=name):
                self.assertEqual(len(puzzle.setup.region_symmetries(puzzle.grid == EMPTY)), 1)
                self.assertEqual(grids(puzzle, up_to_symmetry=True),
                                 grids(puzzle, branching="cell", symmetry=False))

    def test_representatives_regenerate_every_solution(self):
        """Applying the group to the representatives must give back the
        full solution set exactly -- nothing missed, nothing spurious."""
        for name in ("11", "17", "22", "35"):
            puzzle = self.game.books["main_puzzles"][name]
            with self.subTest(puzzle=name):
                images = puzzle.setup.region_symmetries(puzzle.grid == EMPTY)
                reps = [s.grid for s in Solver(puzzle).solve(up_to_symmetry=True)]
                regenerated = {apply_image(g, im).tobytes() for g in reps for im in images}
                self.assertEqual(sorted(regenerated),
                                 grids(puzzle, branching="cell", symmetry=False))

    def test_representatives_are_pairwise_distinct_classes(self):
        for name in ("11", "17", "22"):
            puzzle = self.game.books["main_puzzles"][name]
            with self.subTest(puzzle=name):
                images = puzzle.setup.region_symmetries(puzzle.grid == EMPTY)
                reps = [s.grid for s in Solver(puzzle).solve(up_to_symmetry=True)]
                for a, b in itertools.combinations(reps, 2):
                    for im in images:
                        self.assertFalse(np.array_equal(apply_image(a, im), b),
                                         "two representatives are the same solution")


class TestSelfSymmetricSolutions(unittest.TestCase):
    """The case the orbit code is most likely to double-yield: a puzzle whose
    open region is symmetric but whose only solution is its own mirror image,
    so its orbit has size one."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzlerPRO")

    def test_still_exactly_one_solution(self):
        for name in ("81", "82", "90"):
            puzzle = self.game.books["pyramid_puzzles"][name]
            with self.subTest(puzzle=name):
                self.assertGreater(
                    len(puzzle.setup.region_symmetries(puzzle.grid == EMPTY)), 1,
                    "fixture no longer has a symmetric open region",
                )
                for mode in MODES:
                    self.assertEqual(len(grids(puzzle, **mode)), 1, f"{mode}")
                self.assertEqual(len(grids(puzzle, up_to_symmetry=True)), 1)


class TestPentominoes(unittest.TestCase):
    """Published counts, independent of this solver: the 12 pentominoes tile
    3x20 in 2 ways up to the rectangle's symmetry, i.e. 8 counting all four
    orientations.

    Only 3x20 runs here -- it is the one that fits the suite's time budget.
    4x15 (1472 / 368), 5x12 (4040 / 1010) and 6x10 (9356 / 2339) are the
    same check at minutes rather than seconds, worth running by hand after
    touching the orbit code."""

    CASES = [(20, 3, 8, 2)]

    def test_counts(self):
        for width, depth, total, distinct in self.CASES:
            setup = pentomino_setup(width, depth)
            state = State(setup)
            with self.subTest(board=f"{depth}x{width}"):
                self.assertEqual(len(setup.board_symmetries), 4)
                baseline = grids(state, branching="cell", symmetry=False)
                self.assertEqual(len(baseline), total)
                self.assertEqual(grids(state), baseline, "default mode disagrees")
                self.assertEqual(len(grids(state, up_to_symmetry=True)), distinct)


class TestLimitsWithSymmetry(unittest.TestCase):
    """max_solutions counts derived solutions like any other, and an early
    stop must still leave the solver usable."""

    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")
        cls.empty = cls.game.puzzles["empty_main"]

    def test_max_solutions_is_exact(self):
        for mode in MODES + [{"up_to_symmetry": True}]:
            with self.subTest(**mode):
                for n in (0, 1, 2, 3, 5, 7):
                    found = list(Solver(self.empty).solve(seed=0, max_solutions=n, **mode))
                    self.assertEqual(len(found), n)
                    self.assertEqual(len({s.grid.tobytes() for s in found}), n, "duplicates")

    def test_solutions_are_valid_and_distinct(self):
        found = list(Solver(self.empty).solve(seed=0, max_solutions=25))
        self.assertEqual(len({s.grid.tobytes() for s in found}), 25)
        for state in found:
            assert_valid_solution(self, self.empty, state)

    def test_early_stop_leaves_solver_reusable(self):
        solver = Solver(self.empty)
        first = [s.grid.tobytes() for s in solver.solve(max_solutions=6)]
        again = [s.grid.tobytes() for s in solver.solve(max_solutions=6)]
        self.assertEqual(first, again)

    def test_seeds_give_same_solution_set(self):
        puzzle = self.game.books["main_puzzles"]["17"]
        sets = [grids(puzzle, seed=seed) for seed in (None, 0, 1)]
        for other in sets[1:]:
            self.assertEqual(other, sets[0])

    def test_bad_branching_rejected(self):
        with self.assertRaises(ValueError):
            list(Solver(self.empty).solve(branching="nonsense", max_solutions=1))


if __name__ == "__main__":
    unittest.main()
