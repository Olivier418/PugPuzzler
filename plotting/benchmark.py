import time
import multiprocessing
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from classes import Puzzle, SolutionBook, Solution
from classes._utils import _next_free_idx_dir
from classes.solutions import SolveStats, SolveStatsBook
from serialization import save_solution_run, load_solution_run
from constants import CONFIG_PALETTE, BENCHMARK_DIR


# A config is the dict of keyword options handed to the solver (besides the
# seed). The solver currently has none, so the default is a single empty
# config; benchmarking a solver variant means adding its options here.
DEFAULT_CONFIGS = [{}]

ConfigKey = tuple[tuple[str, object], ...]


def _config_key(options: dict) -> ConfigKey:
    """Hashable identity of a config, used to group trials."""
    return tuple(sorted(options.items()))


def _config_label(options: dict) -> str:
    """Human-readable, filesystem-safe name of a config: its options joined
    by "+" (`k` for a True flag, `k=v` otherwise); "default" if there are
    none. Every option is shown, False ones included, so configs that
    differ only in a switched-off flag still get different labels."""
    parts = [k if v is True else f"{k}={v}" for k, v in sorted(options.items())]
    return "+".join(parts) if parts else "default"


def _worker(puzzle: Puzzle, config: dict, seed: int, T: float, conn):
    """Runs in a subprocess so a runaway search (e.g. every flag off on a
    puzzle with a huge search space) can be killed on a wall-clock
    deadline -- something a plain generator loop in the main process
    can't do safely.

    Deliberately does NOT time itself or check against T: spawning a
    subprocess (reimporting numpy/matplotlib/etc. in the fresh
    interpreter, and unpickling `puzzle`) takes real time that has
    nothing to do with the solver. The parent owns T and stamps arrival
    time itself instead; this just sends a "ready" handshake once that
    startup cost is already behind it (i.e. once this line actually
    starts running), so the parent knows exactly when to start its
    clock, then streams solutions as they're found, and gets
    hard-killed by the parent once T is actually up. The solver is also
    handed `time_limit=T` (counted from its own first node), so it
    normally stops by itself with a clean "done" a hair after the parent's
    deadline; the kill is only the backstop.

    Sends back only the raw grid per solution -- the parent pairs each
    with its own arrival-time stamp to build the SolveStats.
    """
    try:
        conn.send(("ready", None))
        for sol in puzzle.solve(disp=False, seed=seed, time_limit=T, **config):
            # sol.grid is compact; Solution.grids (and the JSON they get
            # saved as) are full board-shaped, same as Solution.from_puzzle.
            conn.send(("solution", puzzle.setup.expand(sol.grid)))
        conn.send(("done", None))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def _run_single_test(puzzle: Puzzle, config: dict, seed: int, T: float) -> tuple[SolutionBook, SolveStatsBook]:
    """One (config, seed) trial, capped at T seconds. Returns a
    (SolutionBook, SolveStatsBook) pair, each containing exactly one
    entry for this trial -- reusing the book containers for a single
    puzzle is a bit unusual, but it's what buys us free
    save_solution_run/load_solution_run compatibility, rather than
    duplicating that tracking in a benchmark-only class.
    """
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_worker, args=(puzzle, config, seed, T, child_conn))

    grids: list[np.ndarray] = []
    elapsed: list[float] = []

    p.start()
    child_conn.close()

    # Block until the worker's "ready" handshake, i.e. until subprocess
    # spawn + module reimport + unpickling `puzzle` is behind it -- only
    # then does the clock start, so that startup cost (the ~1s of
    # apparent "dead time" before the first solution) isn't mistaken for
    # search time. Capped generously (well beyond T) so a worker that
    # dies before ever sending anything can't hang the benchmark.
    SETUP_TIMEOUT = max(T, 5.0) + 30.0
    setup_error = None
    if parent_conn.poll(SETUP_TIMEOUT):
        try:
            status, payload = parent_conn.recv()
            if status != "ready":
                setup_error = payload if status == "error" else f"unexpected message {status!r} before ready"
        except EOFError:
            setup_error = "worker closed its connection before becoming ready"
    else:
        setup_error = f"worker did not become ready within {SETUP_TIMEOUT}s"

    # The single authoritative clock, started only once the worker is
    # actually ready: both the T deadline below and every solution's
    # elapsed time are measured against this, so they can't drift apart
    # the way the parent/worker clocks did before.
    start_wait = time.perf_counter()

    config_label = _config_label(config)
    if setup_error is not None:
        print(f"\nWorker error in config {config_label}, seed {seed}: {setup_error}")
    else:
        while True:
            if not p.is_alive() and not parent_conn.poll():
                break

            rem_time = T - (time.perf_counter() - start_wait)
            if rem_time <= 0:
                break

            if not parent_conn.poll(rem_time):
                break

            try:
                status, payload = parent_conn.recv()
            except EOFError:
                break

            if status == "solution":
                grids.append(payload)
                elapsed.append(time.perf_counter() - start_wait)
            elif status == "done":
                break
            elif status == "error":
                print(f"\nWorker error in config {config_label}, seed {seed}: {payload}")
                break

    duration = time.perf_counter() - start_wait

    if p.is_alive():
        p.terminate()
        p.join()
    parent_conn.close()

    source = puzzle.source
    solution = Solution(
        puzzle_name=puzzle.name,
        grids=grids,
        game_name=source.game_name if source else None,
        book_name=source.book_name if source else None,
        puzzle=puzzle,
    )
    stats = SolveStats(
        puzzle_name=puzzle.name,
        options=dict(config),
        seed=seed,
        duration=duration,
        elapsed=elapsed,
    )
    solution_book = SolutionBook(solution, game_name=solution.game_name, book_name=solution.book_name)
    stats_book = SolveStatsBook(stats, options=config, seed=seed)
    return solution_book, stats_book


def run_benchmark(
    puzzle: Puzzle,
    configs: list[dict] = DEFAULT_CONFIGS,
    nr_tests: int = 10,
    T: float = 5.0,
    base_folder: str | Path = BENCHMARK_DIR,
) -> tuple[dict[ConfigKey, list[SolveStatsBook]], Path]:
    """Run nr_tests trials (seeds 0..nr_tests-1) of each config on puzzle,
    each capped at T seconds. Every trial is saved as its own
    solutions.json + stats.json pair under
    base_folder/<game>/<books|puzzles>/<puzzle-or-book_puzzle>/benchmark_<idx>/
    -- mirroring where the puzzle itself lives under games/, same as
    solutions/ does -- with one benchmark_<idx> folder per call, so
    re-running the same puzzle never overwrites an earlier run and you
    can tell separate simulations apart at a glance.

    Returns the SolveStats side of the results in memory, grouped by config
    (see _config_key), so it can go straight into plot_benchmark without a
    reload -- plus the run
    folder itself, so it can be handed straight to load_benchmark later.
    """
    source = puzzle.source
    puzzle_dir = Path(base_folder) / source.relative_dir() if source else Path(base_folder) / puzzle.name
    puzzle_folder = _next_free_idx_dir(puzzle_dir, prefix="benchmark")

    stats_books: dict[ConfigKey, list[SolveStatsBook]] = {
        _config_key(config): [] for config in configs
    }
    total_tasks = len(configs) * nr_tests

    with tqdm(total=total_tasks, desc="Benchmarking", unit="run") as pbar:
        for config in configs:
            config_folder = puzzle_folder / _config_label(config)

            for seed in range(nr_tests):
                solution_book, stats_book = _run_single_test(puzzle, config, seed, T)
                # flat=False: config/seed folders sit below the puzzle
                # name here, not directly above an idx, so the usual
                # puzzle-name-from-folder trick (see saving.save_solution_run)
                # doesn't apply -- keep puzzle_name explicit in the file.
                save_solution_run(solution_book, stats_book, config_folder / f"test{seed}", flat=False)
                stats_books[_config_key(config)].append(stats_book)
                pbar.update(1)

    return stats_books, puzzle_folder


def load_benchmark(folder: str | Path) -> dict[ConfigKey, list[SolveStatsBook]]:
    """Reload a benchmark previously written by run_benchmark, without
    re-solving anything. `folder` is the run's own folder -- the one
    directly containing each config's subfolder -- i.e. exactly the path
    run_benchmark returned. Mirrors serialization.load_solution_run."""
    puzzle_folder = Path(folder)

    stats_books: dict[ConfigKey, list[SolveStatsBook]] = {}
    for config_folder in sorted(p for p in puzzle_folder.iterdir() if p.is_dir()):
        test_folders = sorted(config_folder.glob("test*"), key=lambda p: int(p.stem.removeprefix("test")))
        for test_folder in test_folders:
            _, stats_book = load_solution_run(test_folder)
            stats_books.setdefault(_config_key(stats_book.options), []).append(stats_book)

    return stats_books


def _solved(stats_book: SolveStatsBook) -> SolveStats:
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
    T: float | None = None,
):
    """`T` is the wall-clock cap the benchmark was run with; pass it to
    pin the x-axis (and the trendlines) to the full window that was
    actually budgeted. Left out, the axis falls back to the longest trial
    observed, which only differs from T when every trial finished early."""
    # B (the plot's time ceiling) has to come from how long each trial
    # actually ran, not from the timestamp of the last solution found --
    # a trial that times out at T can easily go quiet for a while before
    # the kill, so its last *solution* lands well before T even though
    # the search itself ran the full T seconds.
    all_durations = [_solved(book).duration for runs in stats_books.values() for book in runs]
    B = T if T is not None else (max(all_durations) if all_durations else 1.0)

    time_grid = np.linspace(0, B, 500)
    final_max_y = 1

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, (config_key, runs) in enumerate(stats_books.items()):
        c = CONFIG_PALETTE[i % len(CONFIG_PALETTE)]
        config_name = _config_label(dict(config_key))

        grid_counts = []
        fit_runs: list[tuple[np.ndarray, float]] = []

        for book in runs:
            stats = _solved(book)
            duration = min(stats.duration, B)
            times = np.asarray(stats.elapsed, dtype=float)
            times = times[times <= B]

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
                final_max_y = max(final_max_y, rate_estimate * B)

        legend_label = f"{config_name} ({rate_estimate:.2f} solutions/s)"
        ax.plot(time_grid, avg_counts, color=c, alpha=1.0, linewidth=2, label=legend_label)

    ax.set_xlim(0, B)
    ax.set_ylim(0, final_max_y * 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solutions Found")
    ax.set_title("Solutions over Time")
    ax.legend()

    return ax
