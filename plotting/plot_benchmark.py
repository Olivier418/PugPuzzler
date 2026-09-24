import matplotlib.pyplot as plt
import numpy as np

from benchmark import ConfigKey, config_label
from classes import SolveStats, SolveStatsBook
from constants import CONFIG_PALETTE


def _only_stats(stats_book: SolveStatsBook) -> SolveStats:
    """Each book from run_benchmark/load_benchmark holds exactly one
    puzzle (this trial), so unwrap it without caring what it's named."""
    return next(iter(stats_book.values()))


def _poisson_rate(runs: list[tuple[np.ndarray, float]]) -> float:
    """Maximum-likelihood rate of a homogeneous Poisson process observed
    over the trials in `runs`: total solutions divided by total observed
    time, pooled across trials.

    This is the textbook estimator for a counting process, and it answers
    the "what about the gaps?" question by construction -- the quiet
    stretches between solutions are never sampled or fitted, they're
    simply part of the denominator. Every second of observation carries
    equal weight (unlike a least-squares fit to the cumulative curve,
    which weights late time by t), and the resulting line lambda*t passes
    exactly through each trial's endpoint (D, N), so the trendline ends
    where the measured curve ends.

    A trial that exhausted its search space early contributes only its own
    (shorter) duration: we never observed it failing to produce solutions
    after that, because there were none left to produce.
    """
    n_solutions = 0
    total_time = 0.0
    for times, duration in runs:
        if duration <= 0:
            continue
        n_solutions += int(np.count_nonzero(times <= duration))
        total_time += duration
    return n_solutions / total_time if total_time > 0 else 0.0


def plot_benchmark(
    stats_books: dict[ConfigKey, list[SolveStatsBook]],
    show_trendline: bool = True,
):
    # t_max (the plot's time ceiling) has to come from how long each trial
    # actually ran, not from the timestamp of the last solution found --
    # a trial that times out at T can easily go quiet for a while before
    # the kill, so its last *solution* lands well before T even though
    # the search itself ran the full T seconds. The longest duration is
    # the benchmark's T whenever any trial timed out.
    all_durations = [_only_stats(book).duration for runs in stats_books.values() for book in runs]
    t_max = max(all_durations) if all_durations else 1.0

    time_grid = np.linspace(0, t_max, 500)
    final_max_y = 1

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (config_key, runs) in enumerate(stats_books.items()):
        c = CONFIG_PALETTE[i % len(CONFIG_PALETTE)]
        config_name = config_label(dict(config_key))

        grid_counts = []
        fit_runs: list[tuple[np.ndarray, float]] = []

        for book in runs:
            stats = _only_stats(book)
            duration = min(stats.duration, t_max)
            times = np.asarray(stats.elapsed, dtype=float)
            times = times[times <= t_max]

            # N(t) counts solutions found *up to and including* t, so the
            # i-th solution (0-based) takes the curve to i+1, and the curve
            # sits at 0 until the first one -- hence the explicit (0, 0)
            # anchor and the 1-based counts. Getting this off by one is
            # what used to make the plot open at -1 solutions.
            step_x = np.concatenate(([0.0], times, [duration]))
            step_y = np.concatenate(([0.0], np.arange(1, times.size + 1), [times.size]))
            ax.step(step_x, step_y, where='post', color=c, alpha=0.15)

            # side='right' already yields "number of solution times <= t",
            # which is N(t) itself; the old -1 here was the actual bug.
            grid_counts.append(np.searchsorted(times, time_grid, side='right'))
            fit_runs.append((times, duration))

        avg_counts = np.mean(grid_counts, axis=0)
        final_max_y = max(final_max_y, avg_counts.max())

        rate_estimate = 0.0
        if show_trendline:
            rate_estimate = _poisson_rate(fit_runs)
            if rate_estimate > 0:
                # Drawn across the whole window, not just out to the last
                # solution: the trendline is a claim about the rate over
                # the budgeted time, so it should be visible wherever the
                # measured curve is. Because the rate is N/D, this line
                # lands on each trial's final count at that trial's own
                # duration rather than floating above or below it.
                ax.plot(time_grid, rate_estimate * time_grid, color=c, linestyle='--', alpha=0.7, label='_nolegend_')
                final_max_y = max(final_max_y, rate_estimate * t_max)

        legend_label = f"{config_name} ({rate_estimate:.2f} solutions/s)"
        ax.plot(time_grid, avg_counts, color=c, alpha=1.0, linewidth=2, label=legend_label)

    ax.set_xlim(0, t_max)
    ax.set_ylim(0, final_max_y * 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solutions Found")
    ax.set_title("Solutions over Time")
    ax.legend()

    return ax
