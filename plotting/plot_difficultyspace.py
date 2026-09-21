import matplotlib.pyplot as plt
from constants import DIFFICULTY_COLORS, UNKNOWN_DIFFICULTY_COLOR

from classes import SolutionBook
from serialization import solution_puzzle_info


def plot_difficulty_space(solutions: SolutionBook, puzzles=None, ax: plt.Axes = None) -> plt.Axes:
    """Scatter plot: one dot per puzzle. x = empty spaces in the given
    puzzle, y = number of solutions found, colored by difficulty.

    difficulty/empty-cell count are the source Puzzle's, not the
    Solution's own -- see serialization.solution_puzzle_info. Pass `puzzles` (e.g. the
    PuzzleBook `solutions` was solved from) when one is in memory;
    otherwise each is resolved from its puzzle's JSON on disk.
    """
    sols = list(solutions.values())
    infos = [solution_puzzle_info(sol, puzzles) for sol in sols]

    labels = [sol.puzzle_name for sol in sols]
    colors = [DIFFICULTY_COLORS.get(info.difficulty, UNKNOWN_DIFFICULTY_COLOR) for info in infos]
    nr_solutions = [len(sol.grids) for sol in sols]
    nr_empty_spaces = [info.nr_empty_spaces for info in infos]

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))

    ax.scatter(nr_empty_spaces, nr_solutions, c=colors, s=70,
               edgecolors="white", linewidths=0.6, zorder=3)

    for x, y, label in zip(nr_empty_spaces, nr_solutions, labels):
        ax.annotate(label, (x, y), textcoords="offset points",
                    xytext=(6, 4), fontsize=8, color="#444444")

    # minimalist styling
    ax.set_xlabel("Empty spaces in puzzle")
    ax.set_ylabel("Number of solutions")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#666666")
    ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=color,
                   label=difficulty, markersize=8)
        for difficulty, color in DIFFICULTY_COLORS.items()
    ]
    ax.legend(handles=handles, title="Difficulty", frameon=False, loc="best")

    ax.set_title("Puzzle difficulty vs. solution count", fontsize=12, color="#333333")
    plt.tight_layout()
    # set log y scale
    ax.set_yscale("log")
    return ax