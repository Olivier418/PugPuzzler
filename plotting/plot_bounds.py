from collections.abc import Mapping

import matplotlib.pyplot as plt
import numpy as np

from classes import Puzzle, SolveStats
from constants import OUT_OF_BOUNDS_COLOR
from .plot_puzzle_stats import METRICS, format_axes, solved_puzzles


def _bound_curve(puzzles: Mapping[str, Puzzle], stats: Mapping[str, SolveStats]) -> tuple[np.ndarray, np.ndarray]:
    """(filled counts, solution counts) of one bound's puzzles, by filled count."""
    filled, count = METRICS["nr_filled_cells"], METRICS["nr_solutions"]
    points = sorted(
        (filled.value(puzzle, s), count.value(puzzle, s))
        for puzzle, s in zip(solved_puzzles(puzzles, stats), stats.values())
    )
    return np.array([p[0] for p in points], dtype=float), np.array([p[1] for p in points], dtype=float)


def plot_bounds(
    lower_puzzles: Mapping[str, Puzzle],
    lower_stats: Mapping[str, SolveStats],
    upper_puzzles: Mapping[str, Puzzle],
    upper_stats: Mapping[str, SolveStats],
    title: str | None = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """The solution bounds of a board, as puzzle_bounds.compute_puzzle_bounds
    returns them: number of solutions (log) against filled cells, with
    everything outside the bounds shaded (constants.OUT_OF_BOUNDS_COLOR) and
    a dashed black line along each bound, with a small black dot (smaller
    than plot_puzzle_stats' and unnamed) on each of the bounds' own puzzles
    -- the band is the backdrop for a real book.

    Returns the ax, set up exactly as plot_puzzle_stats would set up its own
    (same metrics, labels and styling) but with fixed limits, so overlaying
    any book of the same board on it --
    `plot_puzzle_stats(puzzles, stats, ax=ax)` -- places its puzzles inside
    the band without rescaling the plot."""
    lower_x, lower_y = _bound_curve(lower_puzzles, lower_stats)
    upper_x, upper_y = _bound_curve(upper_puzzles, upper_stats)

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    metrics = (METRICS["nr_filled_cells"], METRICS["nr_solutions"])
    ax._puzzle_stats_metrics = metrics  # what an overlaid plot_puzzle_stats inherits
    format_axes(ax, metrics)

    # Fixed limits (which also switch autoscaling off): the shading below
    # reaches exactly to them, and later overlays can't rescale the plot.
    x_lo, x_hi = min(lower_x[0], upper_x[0]), max(lower_x[-1], upper_x[-1])
    x_pad = max(0.02 * (x_hi - x_lo), 1.0)
    y_lo, y_hi = np.log10(lower_y.min()), np.log10(upper_y.max())
    y_pad = 0.05 * max(y_hi - y_lo, 1.0)
    bottom, top = 10 ** (y_lo - y_pad), 10 ** (y_hi + y_pad)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ax.set_ylim(bottom, top)

    ax.fill_between(lower_x, lower_y, bottom, color=OUT_OF_BOUNDS_COLOR, linewidth=0, zorder=1)
    ax.fill_between(upper_x, upper_y, top, color=OUT_OF_BOUNDS_COLOR, linewidth=0, zorder=1)
    for x, y in ((lower_x, lower_y), (upper_x, upper_y)):
        ax.plot(x, y, linestyle="--", color="black", linewidth=1, marker="o", markersize=4, zorder=2)

    ax.set_title(title or "Solution bounds", fontsize=12, color="#333333")
    ax.figure.tight_layout()
    return ax
