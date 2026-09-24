"""Benchmarking the solver: run a puzzle under several solver configs and
seeds, each trial in a subprocess with a wall-clock cap, save each trial's
SolveStats (never its solutions: a benchmark is about when solutions are
found, not what they are), and load them back. Plotting the results lives
in plotting/plot_benchmark.py."""
import time
import multiprocessing
from collections.abc import Sequence
from pathlib import Path

from tqdm import tqdm

from classes import Puzzle, SolveStats, SolveStatsBook
from constants import BENCHMARK_DIR
from serialization.paths import next_run_dir, puzzle_name_from_run_dir
from serialization.stats_io import load_solve_stats_book, write_solve_stats


# A config is the dict of keyword options handed to the solver (besides the
# seed): `branch` and `order` (see Solver.solve). An empty config means the
# solver's own defaults -- so {"branch": "cell"} is the cells-only baseline,
# not {}. Note that a benchmark trial always sets a time limit, so `order`
# defaults to on.
#
# Only options that leave the solution set alone belong here; anything that
# changes *which* solutions come back (time_limit, max_solutions) would make
# two configs solve different problems, so those are named arguments on
# solve_puzzle instead.
DEFAULT_CONFIGS = (
    {"order": "counts"},
    {"order": "pockets"},
    {"order": "fanout"},
    {"order": False},
)

ConfigKey = tuple[tuple[str, object], ...]


def config_key(options: dict) -> ConfigKey:
    """Hashable identity of a config, used to group trials."""
    return tuple(sorted(options.items()))


def config_label(options: dict) -> str:
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

    Sends back only an empty tick per solution -- the parent pairs each
    with its own arrival-time stamp to build the SolveStats. Pickling the
    grids across the pipe would be billed as search time, for data a
    benchmark never keeps.
    """
    try:
        # Numba's cache load is startup cost too; get it behind "ready".
        puzzle.setup.warmup()
        conn.send(("ready", None))
        for _ in puzzle.solve(seed=seed, time_limit=T, **config):
            conn.send(("solution", None))
        conn.send(("done", None))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def _run_single_test(puzzle: Puzzle, config: dict, seed: int, T: float) -> SolveStats:
    """One (config, seed) trial, capped at T seconds."""
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(target=_worker, args=(puzzle, config, seed, T, child_conn))

    elapsed: list[float] = []

    process.start()
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

    label = config_label(config)
    if setup_error is not None:
        print(f"\nWorker error in config {label}, seed {seed}: {setup_error}")
    else:
        while True:
            if not process.is_alive() and not parent_conn.poll():
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
                elapsed.append(time.perf_counter() - start_wait)
            elif status == "done":
                break
            elif status == "error":
                print(f"\nWorker error in config {label}, seed {seed}: {payload}")
                break

    duration = time.perf_counter() - start_wait

    if process.is_alive():
        process.terminate()
        process.join()
    parent_conn.close()

    return SolveStats(
        puzzle_name=puzzle.name, options=dict(config), seed=seed, duration=duration, elapsed=elapsed,
    )


def run_benchmark(
    puzzle: Puzzle,
    configs: Sequence[dict] = DEFAULT_CONFIGS,
    nr_tests: int = 10,
    T: float = 5.0,
    base_folder: str | Path = BENCHMARK_DIR,
) -> tuple[dict[ConfigKey, list[SolveStats]], Path]:
    """Run nr_tests trials (seeds 0..nr_tests-1) of each config on puzzle,
    each capped at T seconds. Every trial's SolveStats is saved as
    base_folder/<game>/<books|puzzles>/<puzzle-or-book_puzzle>/benchmark_<idx>/<config-label>/stats<seed>.json
    -- mirroring where the puzzle itself lives under games/, same as
    solutions/ does -- with one benchmark_<idx> folder per call, so
    re-running the same puzzle never overwrites an earlier run and you
    can tell separate simulations apart at a glance.

    Returns the trials in memory, grouped by config (see config_key), so
    they can go straight into plot_benchmark without a reload -- plus the
    run folder itself, so it can be handed straight to load_benchmark later.
    """
    # Compile once here, so the trial subprocesses only load the cache.
    puzzle.setup.warmup()

    puzzle_folder = next_run_dir(puzzle.source, base_folder, prefix="benchmark")

    trials: dict[ConfigKey, list[SolveStats]] = {config_key(config): [] for config in configs}
    total_tasks = len(configs) * nr_tests

    with tqdm(total=total_tasks, desc="Benchmarking", unit="run") as pbar:
        for config in configs:
            config_folder = puzzle_folder / config_label(config)
            config_folder.mkdir(parents=True, exist_ok=True)

            for seed in range(nr_tests):
                stats = _run_single_test(puzzle, config, seed, T)
                # Flat: the puzzle's name is recovered from the path on load.
                stats_book = SolveStatsBook(stats, options=stats.options, seed=seed)
                write_solve_stats(stats_book, config_folder / f"stats{seed}.json")
                trials[config_key(config)].append(stats)
                pbar.update(1)

    return trials, puzzle_folder


def load_benchmark(folder: str | Path) -> dict[ConfigKey, list[SolveStats]]:
    """Reload a benchmark previously written by run_benchmark, without
    re-solving anything. `folder` is the run's own folder -- the one
    directly containing each config's subfolder -- i.e. exactly the path
    run_benchmark returned."""
    trials: dict[ConfigKey, list[SolveStats]] = {}
    for config_folder in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        files = sorted(config_folder.glob("stats*.json"), key=lambda p: int(p.stem.removeprefix("stats")))
        for file in files:
            stats_book = load_solve_stats_book(file, puzzle_name_from_run_dir(file))
            stats = next(iter(stats_book.values()))
            trials.setdefault(config_key(stats.options), []).append(stats)
    return trials
