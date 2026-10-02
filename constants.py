from pathlib import Path

EMPTY = -1           # grid cell: on the board, nothing placed there yet
OUTSIDE_BOARD = -2   # grid cell: not part of the board at all
UNPLACED = -1        # chosen_placement_idx: this block hasn't been placed

# Every SolveStats time (duration, elapsed stamps) is rounded to this many
# decimals of a second (1 us) when it is taken, so stats in memory and on disk
# are identical and stats files stay compact; perf_counter noise is far coarser.
TIME_DECIMALS = 6

DIFFICULTY_COLORS = {
    "starter": "#AFC92A",
    "junior": "#FABA1C",
    "expert": "#E11121",
    "master": "#015DA8",
    "wizard": "#8D398E",
}
UNKNOWN_DIFFICULTY_COLOR = "#999999"  # puzzles without a "difficulty" (e.g. the empty boards)
# The colors for master and wizard are actually swapped in the original IQpuzzler game

# puzzle_bounds.py tags the puzzles it finds with HARDEST_DIFFICULTY (the lower
# bound: fewest solutions for their filled-cell count) and EASIEST_DIFFICULTY
# (the upper bound: most), and plotting.plot_puzzle_stats colors those tags
# from BOUND_COLORS -- kept out of DIFFICULTY_COLORS since they aren't the
# game's own difficulty tiers. plotting.plot_bounds shades everything outside
# the bounds in OUT_OF_BOUNDS_COLOR.
HARDEST_DIFFICULTY = "hardest"
EASIEST_DIFFICULTY = "easiest"
# The names of the two bound books (the same on every board).
LOWER_BOUND_BOOK = "lower_bound"
UPPER_BOUND_BOOK = "upper_bound"
BOUND_COLORS = {
    HARDEST_DIFFICULTY: "#000000",
    EASIEST_DIFFICULTY: "#7FD1C7",
}
OUT_OF_BOUNDS_COLOR = "#E6E6E6"

# One color per solver config in plotting.plot_benchmark, handed
# out in the order the configs are plotted (and cycled if there are more).
CONFIG_PALETTE = ["#4FA3D1", "#F76773", "#FFDD87", "#CCD88B"]

# Anchored at the repo root rather than the working directory, so code run
# from elsewhere (tutorial/'s notebook runs from tutorial/) finds the same folders.
ROOT_DIR = Path(__file__).resolve().parent
SOLUTION_DIR = ROOT_DIR / "solutions"
BENCHMARK_DIR = ROOT_DIR / "benchmarks"
BOUNDS_DIR = ROOT_DIR / "bounds"  # per board: its solution bounds, see puzzle_bounds.py
GAMES_DIR = ROOT_DIR / "games"