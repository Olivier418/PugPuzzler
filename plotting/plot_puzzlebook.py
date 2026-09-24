from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

from classes import SolutionBook, Solution, SolveStats, SolveStatsBook
from constants import DIFFICULTY_COLORS, FRONTIER_COLOR, FRONTIER_DIFFICULTY, UNKNOWN_DIFFICULTY_COLOR


class Metric(NamedTuple):
    """One per-puzzle quantity that can go on an axis. `value` gets the
    puzzle's Solution (its Puzzle is `sol.puzzle`) and -- only if
    `needs_stats` -- its SolveStats (None if that book has no entry for
    it), and returns a float; nan means "not defined for this puzzle"
    (e.g. the time to the first solution of an unsolved puzzle) and the
    puzzle is left out."""
    label: str
    value: Callable[[Solution, SolveStats | None], float]
    log: bool = False
    needs_stats: bool = False


def _solution_time(stats: SolveStats | None, i: int) -> float:
    """When the i-th (0 = first, -1 = last) solution was found."""
    return stats.elapsed[i] if stats is not None and stats.elapsed else np.nan


METRICS: dict[str, Metric] = {
    "nr_empty_spaces": Metric("Empty spaces in puzzle", lambda sol, stats: sol.puzzle.nr_empty_spaces),
    "nr_solutions": Metric("Number of solutions", lambda sol, stats: len(sol.grids), log=True),
    "time_to_first_solution": Metric(
        "Time to first solution (s)", lambda sol, stats: _solution_time(stats, 0),
        log=True, needs_stats=True),
    "time_to_last_solution": Metric(
        "Time to last solution (s)", lambda sol, stats: _solution_time(stats, -1),
        log=True, needs_stats=True),
    "solve_time": Metric(
        "Solve time (s)", lambda sol, stats: stats.duration if stats is not None else np.nan,
        log=True, needs_stats=True),
}


def _group_label(names: list[str], max_names: int = 3) -> str:
    """Names of the puzzles sharing one dot: "40, 41, 44", or "40, 41, 44, +3"
    when there are too many to read."""
    label = ", ".join(names[:max_names])
    return f"{label}, +{len(names) - max_names}" if len(names) > max_names else label


def _color(difficulty: str | None, frontier_difficulty: str | None) -> str:
    if frontier_difficulty is not None and difficulty == frontier_difficulty:
        return FRONTIER_COLOR
    return DIFFICULTY_COLORS.get(difficulty, UNKNOWN_DIFFICULTY_COLOR)


def _legend_entries(difficulties: list[str | None], frontier_difficulty: str | None) -> dict[str, str]:
    """difficulty -> color for the `difficulties` present: the game's own
    tiers in DIFFICULTY_COLORS order, then `frontier_difficulty`. Puzzles
    without a difficulty (e.g. the empty boards) get no entry."""
    present = set(difficulties)
    entries = {d: c for d, c in DIFFICULTY_COLORS.items() if d in present}
    if frontier_difficulty in present:
        entries[frontier_difficulty] = FRONTIER_COLOR
    return entries


def _resolve(metric: str | Metric) -> Metric:
    if isinstance(metric, Metric):
        return metric
    if metric not in METRICS:
        raise ValueError(f"Unknown metric {metric!r}; choose from {sorted(METRICS)} or pass a Metric.")
    return METRICS[metric]


def _style_2d(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")
    ax.tick_params(colors="#666666")
    ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)


def _style_3d(ax: plt.Axes) -> None:
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((1, 1, 1, 0))
        axis.pane.set_edgecolor("#CCCCCC")
    ax.tick_params(colors="#666666")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)


def plot_puzzlebook(
    solutions: SolutionBook,
    stats: SolveStatsBook = None,
    x: str | Metric | None = None,
    y: str | Metric | None = None,
    z: str | Metric | None = None,
    frontier_difficulty: str | None = FRONTIER_DIFFICULTY,
    title: str | None = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Scatter plot: one dot per puzzle in `solutions`, colored by
    difficulty. `x`, `y` (default `nr_empty_spaces`, `nr_solutions`) and --
    for a 3D plot -- `z` each name a per-puzzle quantity from METRICS (`nr_empty_spaces`, `nr_solutions`,
    `time_to_first_solution`, `time_to_last_solution`, `solve_time`), or
    are a `Metric` of your own.
    Metrics flagged `log` (counts, times) get a log axis, so their values
    must be positive: a puzzle with no solutions has no time to its first
    one, and one with zero solutions can't sit on a log axis, so both are
    left out. A 3D axis can't be log-scaled, so there the values are
    log10'd and the ticks labeled as powers of ten instead.

    Difficulty and empty-cell count are read off each Solution's Puzzle.
    Timing metrics need `stats`, the SolveStatsBook produced alongside
    `solutions` by the same solve call.

    Puzzles that land on the same point (same values, same color) are
    drawn as one dot labeled with all their names, since they'd otherwise
    hide each other. Near-ties, and ties with an overlaid plot's dots, stay
    as they are.

    `frontier_difficulty` (most_difficult_puzzles' own tag for the puzzles
    it found) is colored via constants.FRONTIER_COLOR instead of
    constants.DIFFICULTY_COLORS, since it isn't one of the game's own
    difficulty tiers.

    Pass an existing `ax` to overlay on it rather than starting a fresh
    plot. The axes are the plot's, not the call's: an `ax` this function
    made already knows its metrics, so an overlay needs no `x`/`y`/`z` (and
    naming a different one raises rather than silently mixing quantities);
    its legend is extended with any difficulty it lacks rather than being
    rebuilt. A 3D `ax` from elsewhere needs `z`.
    """
    given = [None if m is None else _resolve(m) for m in (x, y, z)]
    inherited = getattr(ax, "_puzzlebook_metrics", None)
    if inherited is not None:
        if any(g is not None and g != m for g, m in zip(given, inherited + (None,))):
            raise ValueError(
                "This ax already plots " + ", ".join(m.label for m in inherited)
                + "; an overlay can't switch axes.")
        metrics = list(inherited)
    else:
        metrics = [given[0] or METRICS["nr_empty_spaces"], given[1] or METRICS["nr_solutions"]]
        if given[2] is not None:
            metrics.append(given[2])
    is_3d = len(metrics) == 3
    if stats is None and any(m.needs_stats for m in metrics):
        raise ValueError("A timing metric needs `stats`, the SolveStatsBook solved alongside `solutions`.")

    sols = list(solutions.values())
    difficulties = [sol.puzzle.difficulty for sol in sols]
    columns = [
        np.array([m.value(sol, stats.get(sol.puzzle_name) if stats is not None else None)
                  for sol in sols], dtype=float)
        for m in metrics
    ]

    # drop the puzzles that can't be drawn: an undefined value, or a
    # non-positive one on a log axis
    keep = np.ones(len(sols), dtype=bool)
    for m, column in zip(metrics, columns):
        keep &= np.isfinite(column)
        if m.log:
            keep &= column > 0
    # one dot per distinct (point, color), listing the puzzles on it
    groups: dict[tuple, list[str]] = {}
    for i in np.flatnonzero(keep):
        key = (*(column[i] for column in columns), _color(difficulties[i], frontier_difficulty))
        groups.setdefault(key, []).append(sols[i].puzzle_name)
    labels = [_group_label(names) for names in groups.values()]
    colors = [key[-1] for key in groups]
    columns = [np.array([key[d] for key in groups], dtype=float) for d in range(len(metrics))]
    if is_3d:
        columns = [np.log10(column) if m.log else column for m, column in zip(metrics, columns)]

    if ax is None:
        if is_3d:
            ax = plt.figure(figsize=(8, 6)).add_subplot(projection="3d")
        else:
            _, ax = plt.subplots(figsize=(7, 5))
    elif is_3d != hasattr(ax, "zaxis"):
        raise ValueError("`z` needs a 3D `ax` and a 3D `ax` needs `z`.")
    ax._puzzlebook_metrics = tuple(metrics)

    if is_3d:
        ax.scatter(*columns, c=colors, s=70, edgecolors="white", linewidths=0.6, depthshade=False)
        for point, label in zip(zip(*columns), labels):
            ax.text(*point, f" {label}", fontsize=8, color="#444444")
    else:
        ax.scatter(*columns, c=colors, s=70, edgecolors="white", linewidths=0.6, zorder=3)
        for point, label in zip(zip(*columns), labels):
            ax.annotate(label, point, textcoords="offset points",
                        xytext=(6, 4), fontsize=8, color="#444444")

    # minimalist styling
    axes = [ax.xaxis, ax.yaxis] + ([ax.zaxis] if is_3d else [])
    for axis, metric in zip(axes, metrics):
        axis.set_label_text(metric.label)
        if metric.log and is_3d:
            axis.set_major_locator(MaxNLocator(integer=True))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"$10^{{{v:g}}}$"))
    if is_3d:
        _style_3d(ax)
    else:
        _style_2d(ax)
        if metrics[0].log:
            ax.set_xscale("log")
        if metrics[1].log:
            ax.set_yscale("log")

    # legend: the difficulties present, merged into whatever an overlaid
    # plot already shows. The existing handles/labels live on the Legend
    # artist itself, not on the scatter (which is never given a `label=`).
    existing = ax.get_legend()
    handles = list(existing.legend_handles) if existing is not None else []
    seen = {handle.get_label() for handle in handles}
    handles += [
        Line2D([0], [0], marker="o", linestyle="", color=color, label=difficulty, markersize=8)
        for difficulty, color in _legend_entries(difficulties, frontier_difficulty).items()
        if difficulty not in seen
    ]
    if handles:
        ax.legend(handles=handles, title="Difficulty", frameon=False, loc="best")

    if title or inherited is None:
        ax.set_title(title or " vs. ".join(m.label for m in reversed(metrics)), fontsize=12, color="#333333")
    ax.figure.tight_layout()
    return ax
