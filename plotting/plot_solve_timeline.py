import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from constants import DIFFICULTY_COLORS, UNKNOWN_DIFFICULTY_COLOR

from classes import SolutionBook
from classes.solutions import SolveStatsBook


def plot_solve_timeline(solutions: SolutionBook, stats: SolveStatsBook, puzzles=None, ax: plt.Axes = None) -> plt.Axes:
    """Timeline plot: one column per puzzle, y = time (log scale). A thin
    gray horizontal line marks every solution found; the first solution
    for each puzzle is drawn thicker, in the puzzle's difficulty color.
    Puzzle names (x tick labels) are colored by difficulty, and columns
    are visually grouped into bands by difficulty with small gaps
    between groups. A faint, difficulty-colored rectangle behind each
    column's lines spans the full time the solver ran on that puzzle
    (stats[name].duration) -- not just the window between its first and
    last solution, since the solver may keep searching after the last
    solution was already found.

    Timing (duration/elapsed) comes from `stats`, the SolveStatsBook
    produced alongside `solutions` by the same solve call -- it's
    solver-run metadata, not part of the solutions themselves.
    difficulty is the source Puzzle's, not the Solution's own -- see
    Solution.puzzle_info. Pass `puzzles` (e.g. the PuzzleBook `solutions`
    was solved from) when one is in memory; otherwise each is resolved
    from its puzzle's JSON on disk.
    """
    solved_puzzles = list(solutions.values())
    difficulty_by_name = {sol.puzzle_name: sol.puzzle_info(puzzles).difficulty for sol in solved_puzzles}

    # order columns: group by difficulty (in DIFFICULTY_COLORS order);
    # within a group, preserve the original order the puzzles appear in
    # `solutions` (e.g. the order they were listed in the source JSON)
    # rather than sorting by name -- sorting names as strings would put
    # "10" before "2".
    difficulty_order = {d: i for i, d in enumerate(DIFFICULTY_COLORS)}
    solved_puzzles.sort(key=lambda sol: difficulty_order.get(difficulty_by_name[sol.puzzle_name], 0))

    if ax is None:
        _, ax = plt.subplots(figsize=(0.6 * len(solved_puzzles) + 2, 6))

    col_width = 0.6
    group_gap = 0.6  # extra horizontal space inserted between difficulty groups

    x_positions = []
    x = 0.0
    prev_difficulty = None
    for sol in solved_puzzles:
        difficulty = difficulty_by_name[sol.puzzle_name]
        if prev_difficulty is not None and difficulty != prev_difficulty:
            x += group_gap
        x_positions.append(x)
        prev_difficulty = difficulty
        x += 1.0

    # tiny epsilon so a solution found at elapsed == 0 is still visible
    # on a log-scaled y axis
    all_times = [t for sp in solved_puzzles for t in stats[sp.puzzle_name].elapsed]
    positive_times = [t for t in all_times if t > 0]
    eps = min(positive_times) / 10 if positive_times else 1e-3

    for sol, x in zip(solved_puzzles, x_positions):
        color = DIFFICULTY_COLORS.get(difficulty_by_name[sol.puzzle_name], UNKNOWN_DIFFICULTY_COLOR)
        times = sorted(max(t, eps) for t in stats[sol.puzzle_name].elapsed)

        duration = max(stats[sol.puzzle_name].duration, eps)
        ax.add_patch(Rectangle(
            (x - col_width / 2, eps), col_width, duration - eps,
            facecolor=color, edgecolor="none", alpha=0.15, zorder=1,
        ))

        for i, t in enumerate(times):
            if i == 0:
                ax.hlines(t, x - col_width / 2, x + col_width / 2,
                           color=color, linewidth=2.2, zorder=3)
            else:
                ax.hlines(t, x - col_width / 2, x + col_width / 2,
                           color="#BBBBBB", linewidth=0.8, zorder=2)

    # x ticks: puzzle names, colored by difficulty
    ax.set_xticks(x_positions)
    ax.set_xticklabels([sol.puzzle_name for sol in solved_puzzles],
                        rotation=30, ha="right", rotation_mode="anchor")
    for tick_label, sol in zip(ax.get_xticklabels(), solved_puzzles):
        tick_label.set_color(DIFFICULTY_COLORS.get(difficulty_by_name[sol.puzzle_name], UNKNOWN_DIFFICULTY_COLOR))

    ax.set_xlim(x_positions[0] - 1, x_positions[-1] + 1)

    # minimalist styling, consistent with plot_difficulty_space
    ax.set_yscale("log")
    ax.set_ylabel("Time to solution (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.tick_params(axis="y", colors="#666666")
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)

    handles = [
        plt.Line2D([0], [0], color=color, linewidth=2.2, label=difficulty)
        for difficulty, color in DIFFICULTY_COLORS.items()
    ]
    ax.legend(handles=handles, title="Difficulty (1st solution)", frameon=False,
              loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax.set_title("Solve timeline per puzzle", fontsize=12, color="#333333")
    plt.tight_layout()
    return ax