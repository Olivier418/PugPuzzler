from collections.abc import Mapping

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from pugpuzzler.classes import Puzzle, SolveStats
from pugpuzzler.constants import BOUND_COLORS, DIFFICULTY_COLORS
from .plot_puzzle_stats import difficulty_color, solved_puzzles, style_2d


def plot_solve_timeline(puzzles: Mapping[str, Puzzle], stats: Mapping[str, SolveStats]) -> plt.Figure:
    """Timeline plot: one row per difficulty (the game's tiers, then the
    bounds' tags), one column per puzzle in `stats` order, y = time (log,
    shared). Every solution is a thin gray line, the first a thick one in
    the difficulty's color, over a faint bar spanning the whole solve
    (`duration`). Every row has as many column slots as the largest tier,
    so columns are equal width and left-aligned.

    Difficulty is read off each puzzle, found by name in `puzzles` (any
    mapping covering `stats`); colors are plot_puzzle_stats'."""
    if not stats:
        raise ValueError("plot_solve_timeline needs at least one solved puzzle.")
    difficulty_by_name = {p.name: p.difficulty for p in solved_puzzles(puzzles, stats)}

    difficulty_order = {d: i for i, d in enumerate(DIFFICULTY_COLORS | BOUND_COLORS)}
    rows: dict[str | None, list[str]] = {}
    for name in sorted(stats, key=lambda n: difficulty_order.get(difficulty_by_name[n], len(difficulty_order))):
        rows.setdefault(difficulty_by_name[name], []).append(name)
    n_slots = max(map(len, rows.values()))

    fig, axes = plt.subplots(len(rows), 1, sharey=True, squeeze=False,
                             figsize=(0.45 * n_slots + 2, 1.9 * len(rows) + 0.8))

    # tiny epsilon so a solution found at elapsed == 0 is still visible
    # on a log-scaled y axis
    positive_times = [t for s in stats.values() for t in s.elapsed if t > 0]
    eps = min(positive_times) / 10 if positive_times else 1e-3
    col_width = 0.6

    for ax, (difficulty, names) in zip(axes[:, 0], rows.items()):
        color = difficulty_color(difficulty)
        # plot_puzzle_stats' look, minus what a row of columns doesn't need:
        # the bottom spine, x tick marks and vertical grid lines
        style_2d(ax)
        ax.spines["bottom"].set_visible(False)
        ax.tick_params(axis="x", length=0)
        ax.grid(False, axis="x")

        for x, name in enumerate(names):
            times = sorted(max(t, eps) for t in stats[name].elapsed)
            duration = max(stats[name].duration, eps)
            ax.add_patch(Rectangle(
                (x - col_width / 2, eps), col_width, duration - eps,
                facecolor=color, edgecolor="none", alpha=0.15, zorder=1,
            ))
            # first solution thick in the puzzle's color, the rest thin gray
            xmin, xmax = x - col_width / 2, x + col_width / 2
            ax.hlines(times[1:], xmin, xmax, color="#BBBBBB", linewidth=0.8, zorder=2)
            ax.hlines(times[:1], xmin, xmax, color=color, linewidth=2.2, zorder=3)

        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, color=color)
        ax.set_xlim(-0.7, n_slots - 0.3)
        ax.set_yscale("log")
        ax.set_title(difficulty if difficulty is not None else "no difficulty",
                     loc="left", fontsize=10, color=color)

    axes[0, 0].legend(
        handles=[Line2D([0], [0], color="#555555", linewidth=2.2, label="first solution"),
                 Line2D([0], [0], color="#BBBBBB", linewidth=0.8, label="later solutions")],
        frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=9,
    )
    fig.supylabel("Time to solution (s)", fontsize=10, color="#333333")
    fig.suptitle("Solve timeline per puzzle", fontsize=12, color="#333333")
    fig.tight_layout()
    return fig
