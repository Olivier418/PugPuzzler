"""Regression tests for Solver's symmetry-breaking modes (3 and 4).

Mode 3 breaks symmetry using the whole board's rigid motions, filtered to
those that leave every already-decided cell where it is. Mode 4 instead
derives the *still-open region's own* symmetry group at each node, which
strictly contains mode 3's: it additionally finds symmetric leftover
pockets that no rigid motion of the whole board could produce.

Both modes only ever skip searching a branch by deriving its solutions
from another branch's, so the thing to check everywhere is that the
solution set is bit-for-bit the mode-2 solution set -- no duplicates, none
missing, and every derived solution internally consistent.

Run with `python test_symmetry.py`.
"""

from __future__ import annotations

import itertools
import random
import time

import numpy as np

from classes import Block, BlockCollection, Puzzle, RegularBoard, Setup
from constants import EMPTY, UNPLACED
from serialization import load_game
from Solver import Solver

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_checks = 0


def check(condition, label):
    global _checks
    _checks += 1
    if not condition:
        raise AssertionError(label)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def make_block(coords, letter):
    """A test block. rgb/terminal_color only matter for rendering, which
    none of these tests do."""
    coords = np.asarray(coords, dtype=int)
    coords = coords - coords.min(axis=0)
    return Block(coords, rgb=(0, 0, 0), letter=letter, terminal_color="")


def mask_from_coords(setup, coords):
    """Compact (n_cells,) bool mask marking the given board coordinates."""
    mask = np.zeros(setup.n_cells, dtype=bool)
    flat = np.ravel_multi_index(tuple(np.asarray(coords).T), setup.board.cells.shape)
    mask[setup.flat_to_compact[flat]] = True
    return mask


def solve_set(puzzle, mode, solver_cls=Solver):
    """(solver, {grid bytes}) for one mode. Every solution is checked for
    internal consistency on the way out -- modes 3/4 rewrite
    chosen_placement_idx alongside the grid when they derive a solution by
    transform, and a mismatch there wouldn't show up in the grid set."""
    solver = solver_cls(puzzle)
    grids = set()
    for solution in solver.solve(mode=mode):
        check_solution_consistent(solution)
        grids.add(solution.grid.tobytes())
    return solver, grids


def check_solution_consistent(solution):
    """grid and chosen_placement_idx must agree, and the board must be full."""
    check(not np.any(solution.grid == EMPTY), "solution left an empty cell")
    for block_idx, placement_idx in solution.chosen_placement_idx.items():
        check(placement_idx != UNPLACED, f"block {block_idx} left unplaced")
        cells = solution.placements[block_idx][placement_idx]
        check(
            np.all(solution.grid[cells] == block_idx),
            f"block {block_idx}'s chosen placement doesn't match the grid",
        )


class CountingSolver(Solver):
    """Counts search work (placements actually tried) against symmetry work
    (solutions produced by transforming another branch's)."""

    def __init__(self, game):
        super().__init__(game)
        self.derived = 0
        self.placed = 0
        inner = self.game.place_unchecked

        def counting_place(block_idx, placement_idx):
            self.placed += 1
            return inner(block_idx, placement_idx)

        self.game.place_unchecked = counting_place

    def _apply_symmetry(self, solution, image):
        self.derived += 1
        return super()._apply_symmetry(solution, image)


# --------------------------------------------------------------------------
# 1. Setup.region_symmetries -- pure geometry
# --------------------------------------------------------------------------

def test_region_symmetries_geometry():
    game = load_game("games/IQpuzzler")

    for key, setup in game.setups.items():
        # The whole board is just the all-True region.
        whole = setup.region_symmetries(np.ones(setup.n_cells, dtype=bool))
        check(
            len(whole) == len(setup.board_symmetries)
            and all(np.array_equal(a, b) for a, b in zip(whole, setup.board_symmetries)),
            f"{key}: region_symmetries(all) != board_symmetries",
        )

    setup = game.setups["main"]  # 11x5 rectangle
    check(len(setup.board_symmetries) == 4, "11x5 board should have 4 symmetries")
    check(len(game.setups["pyramid"].board_symmetries) == 8, "5x5 pyramid should have 8")

    # Counts are of *distinct permutations of the region's cells*, which is
    # what the solver acts on -- so they can be smaller than the shape's
    # symmetry group. A domino's group has order 4, but its 180 rotation and
    # its long-axis mirror permute the two cells the same way, so
    # region_symmetries collapses them to 2 (and rightly: identical action
    # on cell sets means identical action on placements).
    expected = {
        "single cell": ([(0, 0)], 1),
        "domino": ([(0, 0), (1, 0)], 2),
        "1x3 bar": ([(0, 0), (1, 0), (2, 0)], 2),
        "L-tromino": ([(0, 0), (0, 1), (1, 0)], 2),
        "2x2 square": ([(0, 0), (0, 1), (1, 0), (1, 1)], 8),
        "3x3 square": ([(i, j) for i in range(3) for j in range(3)], 8),
        "S-tetromino": ([(0, 0), (1, 0), (1, 1), (2, 1)], 2),
        "T-tetromino": ([(0, 0), (1, 0), (2, 0), (1, 1)], 2),
        "U-pentomino": ([(0, 0), (0, 1), (1, 0), (2, 0), (2, 1)], 2),
        "X-pentomino": ([(1, 0), (0, 1), (1, 1), (2, 1), (1, 2)], 8),
        "P-pentomino": ([(0, 0), (0, 1), (1, 0), (1, 1), (2, 0)], 1),  # chiral
        "F-pentomino": ([(1, 0), (2, 0), (0, 1), (1, 1), (1, 2)], 1),  # chiral
    }
    for label, (coords, n) in expected.items():
        got = len(setup.region_symmetries(mask_from_coords(setup, coords)))
        check(got == n, f"{label}: expected {n} symmetries, got {got}")

    # Same shape, moved off the origin and into the middle of the board:
    # the group is a property of the shape, not of where it sits.
    corner = setup.region_symmetries(
        mask_from_coords(setup, [(0, 0), (0, 1), (1, 0), (1, 1)])
    )
    inland = setup.region_symmetries(
        mask_from_coords(setup, [(7, 2), (7, 3), (8, 2), (8, 3)])
    )
    check(len(corner) == len(inland) == 8, "a 2x2 pocket has 8 symmetries wherever it is")

    print(f"  region_symmetries: {len(expected) + 5} shape/board checks")


def test_region_symmetries_is_a_group():
    """The orbit argument in Solver._orbits needs the returned images to
    form a group: identity present, closed under composition. Checked on
    the boards themselves and on random sub-regions of them."""
    game = load_game("games/IQpuzzler")
    rng = random.Random(0)
    n_regions = 0

    for setup in game.setups.values():
        identity = np.arange(setup.n_cells)
        regions = [np.ones(setup.n_cells, dtype=bool)]
        for _ in range(40):
            mask = np.zeros(setup.n_cells, dtype=bool)
            mask[rng.sample(range(setup.n_cells), rng.randint(1, setup.n_cells))] = True
            regions.append(mask)

        for region in regions:
            images = setup.region_symmetries(region)
            n_regions += 1
            keys = {im.tobytes() for im in images}
            check(len(keys) == len(images), "region_symmetries returned duplicates")
            check(
                any(np.array_equal(im, identity) for im in images),
                "region_symmetries omitted the identity",
            )
            outside = ~region
            for im in images:
                check(
                    np.array_equal(im[outside], identity[outside]),
                    "an image moved a cell outside the region",
                )
                check(
                    np.array_equal(np.sort(im), identity),
                    "an image is not a permutation",
                )
                for other in images:
                    check(im[other].tobytes() in keys, "images are not closed under composition")

    print(f"  region_symmetries: group axioms hold on {n_regions} regions")


# --------------------------------------------------------------------------
# 2. The case from SYMMETRY_SOLVER_NOTES.md that mode 3 cannot see
# --------------------------------------------------------------------------

def open_pocket_puzzle():
    """A 4x2 board with A/B/C/D preplaced in the left 2x2, leaving a 2x2
    open pocket on the right for a monomino E and an L-tromino F.

    The pocket obviously has a 4-fold symmetry, but no rigid motion of the
    4x2 board permutes it while leaving A,B,C,D alone -- so mode 3 finds
    nothing here and mode 4 finds the full group of 8."""
    blocks = BlockCollection(
        make_block([(0, 0)], "A"),
        make_block([(0, 0)], "B"),
        make_block([(0, 0)], "C"),
        make_block([(0, 0)], "D"),
        make_block([(0, 0)], "E"),
        make_block([(0, 0), (0, 1), (1, 0)], "F"),
    )
    setup = Setup(blocks, RegularBoard(width=4, depth=2))
    # letter_grid is authored (depth, width) -- see Puzzle._initialize_grid.
    letter_grid = [
        ["A", "D", " ", " "],
        ["B", "C", " ", " "],
    ]
    return Puzzle(setup, letter_grid=letter_grid, name="pocket")


def test_open_pocket():
    puzzle = open_pocket_puzzle()
    setup = puzzle.setup
    available = puzzle.grid == EMPTY

    mode3 = Solver(puzzle)._active_symmetries(available)
    mode4 = setup.region_symmetries(available)
    check(len(mode3) == 1, f"mode 3 should see only the identity here, saw {len(mode3)}")
    check(len(mode4) == 8, f"mode 4 should see the pocket's 8 symmetries, saw {len(mode4)}")

    baseline = None
    for mode in (0, 1, 2, 3, 4):
        solver, grids = solve_set(puzzle, mode, CountingSolver)
        if baseline is None:
            baseline = grids
            check(len(grids) == 4, f"expected 4 solutions, got {len(grids)}")
        check(grids == baseline, f"mode {mode} solution set differs from mode 0")
        if mode == 4:
            check(solver.derived == 3, f"mode 4 should derive 3 of the 4, derived {solver.derived}")

    print("  open pocket: mode 3 sees 1 symmetry, mode 4 sees 8; all modes agree (4 solutions)")


# --------------------------------------------------------------------------
# 3. Nested symmetry: the 4x4 / four-quadrant case from the notes
# --------------------------------------------------------------------------

def quadrant_puzzle():
    """A 4x4 board and four distinct 2x2 blocks. The only exact tiling is
    the four quadrants, so there are 4! = 24 labeled solutions -- and
    symmetry is active at several nested depths, which is what makes this
    the test for symmetries compounding down the recursion tree."""
    square = [(0, 0), (0, 1), (1, 0), (1, 1)]
    blocks = BlockCollection(*[make_block(square, LETTERS[i]) for i in range(4)])
    setup = Setup(blocks, RegularBoard(width=4, depth=4))
    return Puzzle(setup, name="quadrants")


def test_nested_symmetry():
    puzzle = quadrant_puzzle()

    baseline = None
    counts = {}
    for mode in (0, 1, 2, 3, 4):
        solver, grids = solve_set(puzzle, mode, CountingSolver)
        if baseline is None:
            baseline = grids
            check(len(grids) == 24, f"expected 24 solutions, got {len(grids)}")
        check(grids == baseline, f"mode {mode} solution set differs from mode 0")
        counts[mode] = (solver.placed, solver.derived)

    check(counts[4][0] < counts[2][0], "mode 4 should try fewer placements than mode 2")
    check(counts[4][1] > 0, "mode 4 should derive some solutions by symmetry")
    print(
        f"  4x4 quadrants: all modes agree (24 solutions); placements tried "
        f"mode2={counts[2][0]} mode3={counts[3][0]} mode4={counts[4][0]}, "
        f"derived by symmetry mode3={counts[3][1]} mode4={counts[4][1]}"
    )


# --------------------------------------------------------------------------
# 4. Randomised small boards -- the broadest guard on the induction
# --------------------------------------------------------------------------

def random_tiling_puzzle(rng, width, depth, n_blocks):
    """A random board tiled by random connected pieces, used as the
    blockset. Guarantees a solvable Setup (the tiling it came from is one
    solution) without hand-authoring shapes."""
    cells = [(i, j) for i in range(width) for j in range(depth)]
    for _ in range(200):
        rng.shuffle(cells)
        groups = [[c] for c in cells[:n_blocks]]
        free = set(cells[n_blocks:])
        while free:
            options = [
                (g, c)
                for g in groups
                for cell in g
                for c in (
                    (cell[0] + 1, cell[1]), (cell[0] - 1, cell[1]),
                    (cell[0], cell[1] + 1), (cell[0], cell[1] - 1),
                )
                if c in free
            ]
            if not options:
                break
            group, cell = rng.choice(options)
            group.append(cell)
            free.discard(cell)
        if not free:
            blocks = BlockCollection(
                *[make_block(g, LETTERS[i]) for i, g in enumerate(groups)]
            )
            setup = Setup(blocks, RegularBoard(width=width, depth=depth))
            return Puzzle(setup, name=f"random{width}x{depth}")
    raise RuntimeError("failed to generate a random tiling")


def test_random_boards():
    rng = random.Random(12345)
    total_solutions = 0
    cases = 0

    for trial in range(30):
        width = rng.randint(2, 4)
        depth = rng.randint(2, 4)
        n_blocks = rng.randint(2, max(2, (width * depth) // 2))
        puzzle = random_tiling_puzzle(rng, width, depth, n_blocks)

        baseline = None
        for mode in (0, 2, 3, 4):
            _, grids = solve_set(puzzle, mode)
            if baseline is None:
                baseline = grids
            check(
                grids == baseline,
                f"random trial {trial} ({width}x{depth}, {n_blocks} blocks): "
                f"mode {mode} differs from mode 0",
            )
        total_solutions += len(baseline)
        cases += 1

    print(f"  random boards: {cases} boards, {total_solutions} solutions, modes 0/2/3/4 identical")


# --------------------------------------------------------------------------
# 5. Real puzzles -- the correctness regression from CLAUDE.md
# --------------------------------------------------------------------------

def test_real_puzzles():
    game = load_game("games/IQpuzzler")
    targets = [
        ("main_puzzles", [str(n) for n in range(40, 61)]),
        ("pyramid_puzzles", [str(n) for n in range(80, 91)]),
    ]

    n_puzzles = 0
    n_solutions = 0
    for book_name, names in targets:
        book = game.books[book_name]
        for name in names:
            puzzle = book.get(name)
            if puzzle is None:
                continue
            baseline = None
            for mode in (2, 3, 4):
                _, grids = solve_set(puzzle, mode)
                if baseline is None:
                    baseline = grids
                check(
                    grids == baseline,
                    f"{book_name}/{name}: mode {mode} differs from mode 2",
                )
            n_puzzles += 1
            n_solutions += len(baseline)

    check(n_puzzles > 0, "no real puzzles were found to test")
    print(f"  real puzzles: {n_puzzles} puzzles, {n_solutions} solutions, modes 2/3/4 identical")


def test_hardest_pyramid():
    """pyramid_puzzles/100 -- the hardest pyramid puzzle and the one with
    the most solutions. Its search tree is deep and symmetry-poor, which
    made it the worst case for mode 4's per-node cost, so it gets its own
    check rather than riding along with the 80-90 range."""
    game = load_game("games/IQpuzzler")
    puzzle = game.books["pyramid_puzzles"]["100"]

    baseline = None
    for mode in (2, 3, 4):
        _, grids = solve_set(puzzle, mode)
        if baseline is None:
            baseline = grids
        check(grids == baseline, f"pyramid_puzzles/100: mode {mode} differs from mode 2")

    print(f"  pyramid_puzzles/100: modes 2/3/4 identical ({len(baseline)} solutions)")


def test_seeded_runs():
    """A seed shuffles placement row order, which _build_placement_lookup
    is rebuilt per solve() to track. Same solution set regardless -- and
    the shuffle must stay private to the solver, since Setup (placements
    included) is shared by reference across every puzzle in the book."""
    game = load_game("games/IQpuzzler")
    book = game.books["main_puzzles"]
    puzzle = book["50"]
    setup = puzzle.setup

    before = {idx: arr.copy() for idx, arr in setup.placements.items()}
    preplaced_before = dict(puzzle.chosen_placement_idx)

    baseline = {s.grid.tobytes() for s in Solver(puzzle).solve(mode=2)}
    for seed in (0, 1, 7):
        for mode in (0, 2, 3, 4):
            solver = Solver(puzzle)
            grids = set()
            for solution in solver.solve(mode=mode, seed=seed):
                check_solution_consistent(solution)
                grids.add(solution.grid.tobytes())
            check(grids == baseline, f"mode {mode} seed {seed} differs from mode 2")

    check(
        all(np.array_equal(before[idx], arr) for idx, arr in setup.placements.items()),
        "a seeded solve mutated the shared Setup's placements",
    )
    check(
        puzzle.chosen_placement_idx == preplaced_before,
        "a seeded solve mutated the source puzzle's chosen_placement_idx",
    )
    # The shared Setup is what every other puzzle in the book indexes into,
    # so a seeded solve of one must leave the rest intact.
    for other in book.values():
        check_placements_match_grid(other)

    print(f"  seeded runs: modes 0/2/3/4 match across 3 seeds ({len(baseline)} solutions); "
          f"shared Setup untouched")


def check_placements_match_grid(puzzle):
    """Every block the puzzle considers already placed must still be where
    its recorded placement index says it is."""
    for block_idx, placement_idx in puzzle.chosen_placement_idx.items():
        if placement_idx == UNPLACED:
            continue
        cells = puzzle.placements[block_idx][placement_idx]
        check(
            np.all(puzzle.grid[cells] == block_idx),
            f"puzzle {puzzle.name}: block {block_idx}'s recorded placement is stale",
        )


# --------------------------------------------------------------------------
# 6. Payoff report (not a correctness check)
# --------------------------------------------------------------------------

def report_payoff(empty_main_cap=300):
    """Search work (placements actually tried) per solution, by mode.

    The preplaced puzzles are here for contrast, not payoff: their board
    symmetry is broken by the first preplaced letter, so there is nothing
    for either symmetry mode to find, and mode 4's per-node sweep is pure
    overhead. The empty board is where the gain is -- capped, since
    solving it outright runs into the millions of solutions.
    `empty_pyramid` is left out entirely: it is slow enough that even a
    small cap would dominate this script's runtime."""
    game = load_game("games/IQpuzzler")
    cases = [
        ("4x4 quadrants", quadrant_puzzle(), None),
        ("open pocket", open_pocket_puzzle(), None),
        ("main_puzzles/50", game.books["main_puzzles"]["50"], None),
        ("pyramid_puzzles/100", game.books["pyramid_puzzles"]["100"], None),
        ("empty_main", game.puzzles["empty_main"], empty_main_cap),
    ]

    modes = (2, 3, 4)
    print(f"  {'case':<22}{'mode':>6}{'placed':>10}{'derived':>10}{'sols':>8}{'secs':>9}")
    for label, puzzle, cap in cases:
        shown = label if cap is None else f"{label} (first {cap})"

        # Repeat interleaved and keep each mode's best: whichever mode ran
        # first would otherwise absorb the cold-start cost, and on a
        # machine doing anything else at all that swamps the difference
        # being measured.
        stats = {}
        for _ in range(2):
            for mode in modes:
                solver = CountingSolver(puzzle)
                solutions = solver.solve(mode=mode)
                if cap is not None:
                    solutions = itertools.islice(solutions, cap)
                start = time.perf_counter()
                n = sum(1 for _ in solutions)
                elapsed = time.perf_counter() - start
                best = stats.get(mode)
                if best is None or elapsed < best[-1]:
                    stats[mode] = (solver.placed, solver.derived, n, elapsed)

        for mode in modes:
            placed, derived, n, elapsed = stats[mode]
            print(
                f"  {shown if mode == modes[0] else '':<22}{mode:>6}{placed:>10}"
                f"{derived:>10}{n:>8}{elapsed:>9.3f}"
            )


if __name__ == "__main__":
    tests = [
        test_region_symmetries_geometry,
        test_region_symmetries_is_a_group,
        test_open_pocket,
        test_nested_symmetry,
        test_random_boards,
        test_real_puzzles,
        test_hardest_pyramid,
        test_seeded_runs,
    ]
    for test in tests:
        print(f"{test.__name__}:")
        test()

    print("\npayoff (informational, not asserted):")
    report_payoff()

    print(f"\nOK -- {_checks} checks passed.")
