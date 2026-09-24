from pathlib import Path

EMPTY = -1           # grid cell: on the board, nothing placed there yet
OUTSIDE_BOARD = -2   # grid cell: not part of the board at all
UNPLACED = -1        # chosen_placement_idx: this block hasn't been placed

DIFFICULTY_COLORS = {
    "starter": "#AFC92A",
    "junior": "#FABA1C",
    "expert": "#E11121",
    "master": "#015DA8",
    "wizard": "#8D398E",
}
UNKNOWN_DIFFICULTY_COLOR = "#999999"  # puzzles without a "difficulty" (e.g. the empty boards)
# The colors for master and wizard are actually swapped in the original IQpuzzler game

# most_difficult_puzzle.py tags the puzzles it finds with FRONTIER_DIFFICULTY,
# and plotting.plot_puzzlebook colors that tag with FRONTIER_COLOR -- both kept
# out of DIFFICULTY_COLORS since it isn't one of the game's own difficulty tiers.
FRONTIER_DIFFICULTY = "inhuman"
FRONTIER_COLOR = "#000000"

# One color per solver config in plotting.benchmark.plot_benchmark, handed
# out in the order the configs are plotted (and cycled if there are more).
CONFIG_PALETTE = ["#4FA3D1", "#F76773", "#FFDD87", "#CCD88B"]

SOLUTION_DIR = Path("solutions")
BENCHMARK_DIR = Path("benchmarks")
GAMES_DIR = Path("games")