import matplotlib.pyplot as plt
import numpy as np

from pugpuzzler.benchmark import ConfigKey, config_label
from pugpuzzler.classes import SolveStats
from pugpuzzler.constants import CONFIG_PALETTE


def _poisson_rate(runs: list[tuple[np.ndarray, float]]) -> float:
    """Maximum-likelihood rate of a homogeneous Poisson process observed
    over the trials in `runs`: total solutions / total observed time,
    pooled across trials. Every second of observation weighs the same, gaps
    between solutions included, and a trial that ran out of solutions
    early counts only its own duration. The line rate*t passes through the
    trials' mean endpoint (mean duration, mean count)."""
    n_solutions = 0
    total_time = 0.0
    for times, duration in runs:
        if duration <= 0:
            continue
        n_solutions += int(np.count_nonzero(times <= duration))
        total_time += duration
    return n_solutions / total_time if total_time > 0 else 0.0


def plot_benchmark(
    trials: dict[ConfigKey, list[SolveStats]],
    show_trendline: bool = True,
):
    """Each trial's cumulative solutions over time (faint steps), each
    config's average (bold) and, with `show_trendline`, its Poisson rate
    (dashed, see _poisson_rate)."""
    # The time axis ends at the longest duration (T if any trial timed out),
    # not the last solution: a search can go quiet long before it stops.
    all_durations = [stats.duration for runs in trials.values() for stats in runs]
    t_max = max(all_durations) if all_durations else 1.0

    time_grid = np.linspace(0, t_max, 500)
    final_max_y = 1

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (config_key, runs) in enumerate(trials.items()):
        c = CONFIG_PALETTE[i % len(CONFIG_PALETTE)]
        config_name = config_label(dict(config_key))

        grid_counts = []
        fit_runs: list[tuple[np.ndarray, float]] = []

        for stats in runs:
            duration = min(stats.duration, t_max)
            times = np.asarray(stats.elapsed, dtype=float)
            times = times[times <= t_max]

            # N(t) = solutions found up to and including t, from (0, 0)
            step_x = np.concatenate(([0.0], times, [duration]))
            step_y = np.concatenate(([0.0], np.arange(1, times.size + 1), [times.size]))
            ax.step(step_x, step_y, where='post', color=c, alpha=0.15)

            # side='right': the number of solution times <= t, i.e. N(t)
            grid_counts.append(np.searchsorted(times, time_grid, side='right'))
            fit_runs.append((times, duration))

        avg_counts = np.mean(grid_counts, axis=0)
        final_max_y = max(final_max_y, avg_counts.max())

        rate_estimate = 0.0
        if show_trendline:
            rate_estimate = _poisson_rate(fit_runs)
            if rate_estimate > 0:
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
