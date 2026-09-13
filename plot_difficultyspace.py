import matplotlib.pyplot as plt
from constants import DIFFICULTY_COLORS, EMPTY

from classes import SolutionBook




def plot_difficulty_space(solutions: SolutionBook, ax: plt.Axes = None) -> plt.Axes:
    """Scatter plot: one dot per puzzle. x = empty spaces in the given
    puzzle, y = number of solutions found, colored by difficulty."""
    solved_puzzles = list(solutions.values())

    labels = [sp.puzzle.name for sp in solved_puzzles]
    colors = [DIFFICULTY_COLORS[sp.puzzle.difficulty] for sp in solved_puzzles]
    nr_solutions = [len(sp.results) for sp in solved_puzzles]
    nr_empty_spaces = [int((sp.puzzle.grid == EMPTY).sum()) for sp in solved_puzzles]

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