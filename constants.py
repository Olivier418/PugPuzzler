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
# The colors for master and wizard are actually swapped in the original IQpuzzler game

MODE_COLORS = {
    0: "#CCD88B",
    1: "#FFDD87",
    2: "#F76773",
    3: "#B06FC4",
    4: "#4FA3D1",
}

SOLUTION_DIR = "solutions"
BENCHMARK_DIR = "benchmarks"
GAMES_DIR = "games"