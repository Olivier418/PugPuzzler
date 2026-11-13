from pathlib import Path

EMPTY = -1           # grid cell: on the board, nothing placed there yet
OUTSIDE_BOARD = -2   # grid cell: not part of the board at all
UNPLACED = -1        # chosen_placement_idx: this block hasn't been placed

# SolveStats times are rounded to 1 us when taken, so memory and disk agree
# exactly and stats files stay compact; perf_counter noise is far coarser.
TIME_DECIMALS = 6

# The game's difficulty tiers (master's and wizard's colors are swapped
# relative to the original IQpuzzler game).
DIFFICULTY_COLORS = {
    "starter": "#AFC92A",
    "junior": "#FABA1C",
    "expert": "#E11121",
    "master": "#015DA8",
    "wizard": "#8D398E",
}
UNKNOWN_DIFFICULTY_COLOR = "#999999"  # puzzles without a "difficulty" (e.g. the empty boards)

# puzzle_bounds.py's tags for the lower bound (fewest solutions) and the upper
# bound (most), colored from BOUND_COLORS since they aren't game tiers.
HARDEST_DIFFICULTY = "hardest"
EASIEST_DIFFICULTY = "easiest"
# The names of the two bound books (the same on every board).
LOWER_BOUND_BOOK = "lower_bound"
UPPER_BOUND_BOOK = "upper_bound"
BOUND_COLORS = {
    HARDEST_DIFFICULTY: "#000000",
    EASIEST_DIFFICULTY: "#7FD1C7",
}
OUT_OF_BOUNDS_COLOR = "#E6E6E6"  # plot_bounds' shading outside the bounds

# plot_benchmark's colors, one per config in plotting order (cycled).
CONFIG_PALETTE = ["#4FA3D1", "#F76773", "#FFDD87", "#CCD88B"]

# Anchored at the repo root rather than the working directory, so code run from
# elsewhere (e.g. tutorial/) finds the same folders. Needs `pip install -e .`.
ROOT_DIR = Path(__file__).resolve().parent.parent
SOLUTION_DIR = ROOT_DIR / "solutions"
BENCHMARK_DIR = ROOT_DIR / "benchmarks"
BOUNDS_DIR = ROOT_DIR / "bounds"  # per board: its solution bounds, see puzzle_bounds.py
GAMES_DIR = ROOT_DIR / "games"