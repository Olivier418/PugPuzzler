import time
import multiprocessing
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from classes import Game, SolutionBook, SolvedPuzzle, PuzzleSetup
from classes.solutions import Result
from serialization import save_solutions, load_solutions
from constants import MODE_COLORS

name_dct = {0: "no pruning / branching", 1: "pruning", 2: "branching"}


def _worker(G: Game, mode: int, seed: int, conn):
    """Runs in a subprocess so a runaway search (e.g. mode=0 on a puzzle
    with a huge search space) can be killed on a wall-clock deadline --
    something a plain generator loop in the main process can't do safely.

    Deliberately does NOT time itself or check against T: spawning a
    subprocess (reimporting numpy/matplotlib/etc. in the fresh
    interpreter) takes real time, so a clock started on the first line
    here already lags the parent's clock by however long that spawn took.
    Enforcing T against this lagging clock would silently cut every run
    short by the spawn delay -- exactly the "data stops before T" bug.
    The parent owns T and stamps arrival time itself instead; this just
    streams solutions as they're found, and gets hard-killed by the
    parent once T is actually up.

    Sends back only the raw grid + chosen_placement_idx per solution, not
    a full Puzzle -- the parent already holds G.setup by reference, so
    reconstructing the solved Puzzle there avoids re-pickling the (shared,
    potentially large) setup on every single message.
    """
    try:
        for sol in G.solve(disp=False, mode=mode, seed=seed):
            conn.send(("solution", (sol.grid, sol.chosen_placement_idx)))
        conn.send(("done", None))
    except Exception as e:
        conn.send(("error", str(e)))
    finally:
        conn.close()


def _run_single_test(G: Game, mode: int, seed: int, T: float) -> SolutionBook:
    """One (mode, seed) trial, capped at T seconds. Returns a SolutionBook
    containing exactly one SolvedPuzzle (this trial) -- reusing SolutionBook
    for a single puzzle is a bit unusual, but it's what buys us `results`
    (with real solutions + elapsed times), `duration`, and free
    save_solutions/load_solutions compatibility, rather than duplicating
    that tracking in a benchmark-only class.
    """
    parent_conn, child_conn = multiprocessing.Pipe()
    p = multiprocessing.Process(target=_worker, args=(G, mode, seed, child_conn))

    results: list[Result] = []

    p.start()
    child_conn.close()
    # The single authoritative clock: both the T deadline below and every
    # solution's elapsed time are measured against this, so they can't
    # drift apart the way the parent/worker clocks did before.
    start_wait = time.perf_counter()

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
            elapsed = time.perf_counter() - start_wait
            grid, chosen_placement_idx = payload
            solution = G.copy()
            solution.grid = grid
            solution.chosen_placement_idx = chosen_placement_idx
            if hasattr(solution, "name"):
                solution.name = f"{G.name} (solution {len(results) + 1})"
            results.append(Result(solution, elapsed))
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

    solved_puzzle = SolvedPuzzle(G, results, duration)
    return SolutionBook(solved_puzzle, mode=mode, seed=seed)


def run_benchmark(
    G: Game,
    modes: list[int] = [0, 1, 2],
    nr_tests: int = 10,
    T: float = 5.0,
    base_folder: str | Path = "benchmarks",
) -> dict[int, list[SolutionBook]]:
    """Run nr_tests trials (seeds 0..nr_tests-1) of each mode on G, each
    capped at T seconds. Every trial is saved as its own SolutionBook json
    under base_folder/G.name/ -- one folder per benchmarked puzzle, so
    benchmarking several puzzles against the same base_folder doesn't
    overwrite anything.

    Returns the same data in memory, grouped by mode, so it can go
    straight into plot_benchmark without a reload. To reuse without
    re-solving later, see load_benchmark.
    """
    puzzle_folder = Path(base_folder) / G.name

    books: dict[int, list[SolutionBook]] = {mode: [] for mode in modes}
    total_tasks = len(modes) * nr_tests

    with tqdm(total=total_tasks, desc="Benchmarking", unit="run") as pbar:
        for mode in modes:
            mode_folder = puzzle_folder / f"mode{mode}"
            mode_folder.mkdir(parents=True, exist_ok=True)

            for seed in range(nr_tests):
                solution_book = _run_single_test(G, mode, seed, T)
                save_solutions(solution_book, mode_folder / f"test{seed}.json")
                books[mode].append(solution_book)
                pbar.update(1)

    return books


def load_benchmark(puzzle_name: str, setup: PuzzleSetup, base_folder: str | Path = "benchmarks") -> dict[int, list[SolutionBook]]:
    """Reload a benchmark previously written by run_benchmark, without
    re-solving anything. Mirrors serialization.load_solutions."""
    puzzle_folder = Path(base_folder) / puzzle_name

    books: dict[int, list[SolutionBook]] = {}
    for mode_folder in sorted(puzzle_folder.glob("mode*")):
        test_files = sorted(mode_folder.glob("test*.json"), key=lambda p: int(p.stem.removeprefix("test")))
        for file_path in test_files:
            book = load_solutions(file_path, setup)
            books.setdefault(book.mode, []).append(book)

    return books


def _solved(book: SolutionBook) -> SolvedPuzzle:
    """Each book from run_benchmark/load_benchmark holds exactly one
    puzzle (this trial), so unwrap it without caring what it's named."""
    return next(iter(book.values()))


def plot_benchmark(books: dict[int, list[SolutionBook]], show_trendline: bool = True):
    # B (the plot's time ceiling) has to come from how long each trial
    # actually ran, not from the timestamp of the last solution found --
    # a trial that times out at T can easily go quiet for a while before
    # the kill, so its last *solution* lands well before T even though
    # the search itself ran the full T seconds.
    all_durations = [_solved(book).duration for runs in books.values() for book in runs]
    B = max(all_durations) if all_durations else 1.0

    time_grid = np.linspace(0, B, 500)
    final_max_y = 1

    fig, ax = plt.subplots(figsize=(8, 5))

    for mode, runs in books.items():
        c = MODE_COLORS.get(mode, "gray")
        mode_name = name_dct.get(mode, f"Mode {mode}")

        interp_runs = []
        mode_max_time = 0.0

        for book in runs:
            elapsed_times = [r.elapsed for r in _solved(book).results]
            trimmed = [t for t in elapsed_times if t <= B]
            counts = np.arange(len(trimmed))
            ax.step(trimmed, counts, where='post', color=c, alpha=0.15)
            final_max_y = max(final_max_y, len(trimmed))
            if trimmed:
                mode_max_time = max(mode_max_time, trimmed[-1])

            counts_at_grid = np.searchsorted(trimmed, time_grid, side='right') - 1
            interp_runs.append(counts_at_grid)

        avg_counts = np.mean(interp_runs, axis=0)

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

        legend_label = f"{mode_name} ({rate_estimate:.2f} solutions/s)"
        ax.plot(time_grid, avg_counts, color=c, alpha=1.0, linewidth=2, label=legend_label)

    ax.set_xlim(0, B)
    ax.set_ylim(0, final_max_y * 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Solutions Found")
    ax.set_title("Solutions over Time")
    ax.legend()

    return ax