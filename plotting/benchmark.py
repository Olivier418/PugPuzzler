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
from constants import MODE_COLORS, BENCHMARK_DIR

name_dct = {0: "no pruning / branching", 1: "pruning", 2: "branching"}


def _worker(puzzle: Puzzle, mode: int, seed: int, conn):
    """Runs in a subprocess so a runaway search (e.g. mode=0 on a puzzle
    with a huge search space) can be killed on a wall-clock deadline --
    something a plain generator loop in the main process can't do safely.

    Deliberately does NOT time itself or check against T: spawning a
    subprocess (reimporting numpy/matplotlib/etc. in the fresh
    interpreter, and unpickling `puzzle`) takes real time that has
    nothing to do with the solver. The parent owns T and stamps arrival
    time itself instead; this just sends a "ready" handshake once that
    startup cost is already behind it (i.e. once this line actually
    starts running), so the parent knows exactly when to start its
    clock, then streams solutions as they're found, and gets
    hard-killed by the parent once T is actually up.

    Sends back only the raw grid per solution -- the parent pairs each
    with its own arrival-time stamp to build the SolveStats.
    """
    try:
        conn.send(("ready", None))
        for sol in puzzle.solve(disp=False, mode=mode, seed=seed):
            # sol.grid is compact; Solution.grids (and the JSON they get
            # saved as) are full board-shaped, same as Solution.from_puzzle.
            conn.send(("solution", puzzle.setup.expand(sol.grid)))
        conn.send(("done", None))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def _run_single_test(puzzle: Puzzle, mode: int, seed: int, T: float) -> tuple[SolutionBook, SolveStatsBook]:
    """One (mode, seed) trial, capped at T seconds. Returns a
    (SolutionBook, SolveStatsBook) pair, each containing exactly one
    entry for this trial -- reusing the book containers for a single
    puzzle is a bit unusual, but it's what buys us free
    save_solution_run/load_solution_run compatibility, rather than
    duplicating that tracking in a benchmark-only class.
    """
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_worker, args=(puzzle, mode, seed, child_conn))

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

    if setup_error is not None:
        print(f"\nWorker error in mode {mode}, seed {seed}: {setup_error}")
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
                print(f"\nWorker error in mode {mode}, seed {seed}: {payload}")
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
        mode=mode,
        seed=seed,
        duration=duration,
        elapsed=elapsed,
    )
    solution_book = SolutionBook(solution, game_name=solution.game_name, book_name=solution.book_name)
    stats_book = SolveStatsBook(stats, mode=mode, seed=seed)
    return solution_book, stats_book


def run_benchmark(
    puzzle: Puzzle,
    modes: list[int] = [0, 1, 2],
    nr_tests: int = 10,
    T: float = 5.0,
    base_folder: str | Path = BENCHMARK_DIR,
) -> tuple[dict[int, list[SolveStatsBook]], Path]:
    """Run nr_tests trials (seeds 0..nr_tests-1) of each mode on puzzle,
    each capped at T seconds. Every trial is saved as its own
    solutions.json + stats.json pair under
    base_folder/<game>/<books|puzzles>/<puzzle-or-book_puzzle>/benchmark_<idx>/
    -- mirroring where the puzzle itself lives under games/, same as
    solutions/ does -- with one benchmark_<idx> folder per call, so
    re-running the same puzzle never overwrites an earlier run and you
    can tell separate simulations apart at a glance.

    Returns the SolveStats side of the results in memory, grouped by
    mode, so it can go straight into plot_benchmark without a reload --
    plus the run folder itself, so it can be handed straight to
    load_benchmark later.
    """
    source = puzzle.source
    puzzle_dir = Path(base_folder) / source.relative_dir() if source else Path(base_folder) / puzzle.name
    puzzle_folder = _next_free_idx_dir(puzzle_dir, prefix="benchmark")

    stats_books: dict[int, list[SolveStatsBook]] = {mode: [] for mode in modes}
    total_tasks = len(modes) * nr_tests

    with tqdm(total=total_tasks, desc="Benchmarking", unit="run") as pbar:
        for mode in modes:
            mode_folder = puzzle_folder / f"mode{mode}"

            for seed in range(nr_tests):
                solution_book, stats_book = _run_single_test(puzzle, mode, seed, T)
                # flat=False: mode/seed folders sit below the puzzle name
                # here, not directly above an idx, so the usual
                # puzzle-name-from-folder trick (see saving.save_solution_run)
                # doesn't apply -- keep puzzle_name explicit in the file.
                save_solution_run(solution_book, stats_book, mode_folder / f"test{seed}", flat=False)
                stats_books[mode].append(stats_book)
                pbar.update(1)

    return stats_books, puzzle_folder


def load_benchmark(folder: str | Path) -> dict[int, list[SolveStatsBook]]:
    """Reload a benchmark previously written by run_benchmark, without
    re-solving anything. `folder` is the run's own folder -- the one
    directly containing mode0/, mode1/, ... -- i.e. exactly the path
    run_benchmark returned. Mirrors serialization.load_solution_run."""
    puzzle_folder = Path(folder)

    stats_books: dict[int, list[SolveStatsBook]] = {}
    for mode_folder in sorted(puzzle_folder.glob("mode*")):
        test_folders = sorted(mode_folder.glob("test*"), key=lambda p: int(p.stem.removeprefix("test")))
        for test_folder in test_folders:
            _, stats_book = load_solution_run(test_folder)
            stats_books.setdefault(stats_book.mode, []).append(stats_book)

    return stats_books


def _solved(stats_book: SolveStatsBook) -> SolveStats:
    """Each book from run_benchmark/load_benchmark holds exactly one
    puzzle (this trial), so unwrap it without caring what it's named."""
    return next(iter(stats_book.values()))


def plot_benchmark(stats_books: dict[int, list[SolveStatsBook]], show_trendline: bool = True):
    # B (the plot's time ceiling) has to come from how long each trial
    # actually ran, not from the timestamp of the last solution found --
    # a trial that times out at T can easily go quiet for a while before
    # the kill, so its last *solution* lands well before T even though
    # the search itself ran the full T seconds.
    all_durations = [_solved(book).duration for runs in stats_books.values() for book in runs]
    B = max(all_durations) if all_durations else 1.0

    time_grid = np.linspace(0, B, 500)
    final_max_y = 1

    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, runs in stats_books.items():
        c = MODE_COLORS.get(mode, "gray")
        mode_name = name_dct.get(mode, f"Mode {mode}")

        interp_runs = []
        mode_max_time = 0.0

        for book in runs:
            elapsed_times = _solved(book).elapsed
            trimmed = [t for t in elapsed_times if t <= B]
            counts = np.arange(len(trimmed))
            ax.step(trimmed, counts, where='post', color=c, alpha=0.15)
            if trimmed:
                mode_max_time = max(mode_max_time, trimmed[-1])

            counts_at_grid = np.searchsorted(trimmed, time_grid, side='right') - 1
            interp_runs.append(counts_at_grid)

        avg_counts = np.mean(interp_runs, axis=0)
        final_max_y = max(final_max_y, avg_counts.max())

        rate_estimate = 0.0
        if show_trendline and mode_max_time > 0:
            active_idx = time_grid <= mode_max_time
            x, y = time_grid[active_idx], avg_counts[active_idx]
            if x.size > 1 and np.sum(x * x) > 0:
                # Least-squares fit forced through the origin: at t=0 there
                # are always 0 solutions found, so the trendline should
                # start there instead of wherever an unconstrained
                # intercept happens to land.
                rate_estimate = np.sum(x * y) / np.sum(x * x)
                ax.plot(x, rate_estimate * x, color=c, linestyle='--', alpha=0.7, label='_nolegend_')
                # The trendline is a straight line from the origin, so its
                # peak is just its value at the right edge of its own
                # active range -- no need to sample it.
                final_max_y = max(final_max_y, rate_estimate * mode_max_time)

        legend_label = f"{mode_name} ({rate_estimate:.2f} solutions/s)"
        ax.plot(time_grid, avg_counts, color=c, alpha=1.0, linewidth=2, label=legend_label)

    ax.set_xlim(0, B)
    ax.set_ylim(0, final_max_y * 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solutions Found")
    ax.set_title("Solutions over Time")
    ax.legend()

    return ax
