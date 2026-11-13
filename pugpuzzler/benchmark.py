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

from pugpuzzler.classes import Puzzle, SolveStats
from pugpuzzler.constants import BENCHMARK_DIR, TIME_DECIMALS
from pugpuzzler.serialization import load_stats, next_run_dir, write_stats


# A config is the dict of solver keyword options besides the seed (`branch`,
# `order`; see solver.solve_rows); {} means the solver's defaults. A trial
# always sets a time limit, so `order` defaults to on. Only options that leave
# the solution set alone belong here (not time_limit/max_solutions).
DEFAULT_CONFIGS = (
    {"order": True},
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


# The worker sends its solution stamps in batches at most this often: one
# pipe message per solution cost ~20% of the search's throughput.
_BATCH_SECONDS = 0.01

# How long past T the parent waits for the worker's final batch and "done"
# before killing it. The worker's own time_limit=T normally ends it a chunk
# (~2 ms) after T.
_GRACE_SECONDS = 1.0


def _worker(puzzle: Puzzle, config: dict, seed: int, T: float, conn):
    """One trial, in a subprocess so a runaway search can be killed on a
    wall-clock deadline.

    Startup (spawn, imports, unpickling `puzzle`, warmup) is kept off the
    clock: the worker sends "ready" and times from then, the moment the
    parent starts T. It solves with Puzzle.solve_rows, as timed_solve does,
    and sends only the solution stamps, ("solutions", [elapsed, ...]) at
    most every _BATCH_SECONDS, then ("done", (duration, complete)). It is
    given `time_limit=T`, so the parent's kill is only the backstop."""
    try:
        puzzle.setup.warmup()
        conn.send(("ready", None))
        start = last_send = time.perf_counter()
        batch = []
        solutions = puzzle.solve_rows(seed=seed, time_limit=T, **config)
        while True:
            try:
                next(solutions)
            except StopIteration as done:  # a for loop would drop its value
                complete = done.value
                break
            now = time.perf_counter()
            batch.append(now - start)
            if now - last_send >= _BATCH_SECONDS:
                conn.send(("solutions", batch))
                batch, last_send = [], now
        duration = time.perf_counter() - start
        conn.send(("solutions", batch))
        conn.send(("done", (duration, complete)))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def _run_single_test(puzzle: Puzzle, config: dict, seed: int, T: float) -> SolveStats:
    """One (config, seed) trial, capped at T seconds. The parent owns the
    deadline: it starts T at the worker's "ready", keeps only solutions
    stamped within T, and kills a worker still running _GRACE_SECONDS past
    it."""
    parent_conn, child_conn = multiprocessing.Pipe()
    process = multiprocessing.Process(target=_worker, args=(puzzle, config, seed, T, child_conn))

    elapsed: list[float] = []
    worker_duration = worker_complete = None

    process.start()
    child_conn.close()

    # Wait for "ready" (startup off the clock), capped so a worker that dies
    # before sending anything can't hang the benchmark.
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

    # The parent's clock, started at the same moment as the worker's.
    start_wait = time.perf_counter()
    deadline = start_wait + T + _GRACE_SECONDS

    label = config_label(config)
    if setup_error is not None:
        print(f"\nWorker error in config {label}, seed {seed}: {setup_error}")
    else:
        while True:
            rem_time = deadline - time.perf_counter()
            if rem_time <= 0 or not parent_conn.poll(rem_time):
                break
            try:
                status, payload = parent_conn.recv()
            except EOFError:  # the worker died without saying goodbye
                break
            if status == "solutions":
                elapsed.extend(payload)
            elif status == "done":
                worker_duration, worker_complete = payload
                break
            elif status == "error":
                print(f"\nWorker error in config {label}, seed {seed}: {payload}")
                break

    stopped = time.perf_counter() - start_wait

    if process.is_alive():
        process.terminate()
        process.join()
    parent_conn.close()

    # The worker's own stop overshoots T by up to a solver chunk; a trial is
    # exactly T long, and complete only if the search ended within it.
    complete = bool(worker_complete) and all(e <= T for e in elapsed) and worker_duration <= T
    elapsed = [round(e, TIME_DECIMALS) for e in elapsed if e <= T]
    duration = round(min(worker_duration if worker_duration is not None else stopped, T), TIME_DECIMALS)

    return SolveStats(
        puzzle_name=puzzle.name, options=dict(config), seed=seed, duration=duration, complete=complete,
        elapsed=elapsed,
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
    <base_folder>/<Source.relative_dir()>/benchmark_<idx>/<config label>/stats<seed>.json,
    a new benchmark_<idx> per call.

    Returns the trials grouped by config (see config_key), ready for
    plot_benchmark, and the benchmark_<idx> folder, for load_benchmark."""
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
                write_stats({puzzle.name: stats}, config_folder / f"stats{seed}.json")
                trials[config_key(config)].append(stats)
                pbar.update(1)

    return trials, puzzle_folder


def load_benchmark(folder: str | Path) -> dict[ConfigKey, list[SolveStats]]:
    """The trials run_benchmark saved in `folder` (the benchmark_<idx> folder
    it returned), grouped by config in config-folder order, each group in
    seed order."""
    trials: dict[ConfigKey, list[SolveStats]] = {}
    for file in sorted(Path(folder).glob("*/stats*.json")):
        for stats in load_stats(file).values():
            trials.setdefault(config_key(stats.options), []).append(stats)
    return {key: sorted(runs, key=lambda s: s.seed) for key, runs in trials.items()}
