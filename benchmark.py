import os
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import multiprocessing
from tqdm import tqdm
import pickle
from classes import Game

name_dct = {0:"no pruning / branching", 1: "pruning", 2:"branching"}

def compare(G: Game, modes: list = [0, 1, 2], nr_tests: int = 10):
    runtimes = np.empty((len(modes), nr_tests), dtype=float)
    solutions_found = np.empty((len(modes), nr_tests), dtype=int)
    
    total_tasks = len(modes) * nr_tests
    with tqdm(total=total_tasks, desc="Comparing Modes", unit="run") as pbar:

        for test_idx in range(nr_tests):
            for i, mode in enumerate(modes):

                start = time.perf_counter()
                # Consume generator and count outputs
                count = sum(1 for _ in G.solve(disp=False, mode=mode, seed=test_idx))
                end = time.perf_counter()
                
                runtimes[i, test_idx] = end - start
                solutions_found[i, test_idx] = count

                pbar.update(1)

    # Calculate statistics
    avg_sol = np.mean(solutions_found, axis=1)
    avg_time = np.mean(runtimes, axis=1)
    std_time = np.std(runtimes, axis=1)
    
    # Avoid division by zero for avg_time_per_sol
    with np.errstate(divide='ignore', invalid='ignore'):
        avg_time_per_sol = avg_time / avg_sol

    # Print Table
    print(f"{'Mode':<25} | {'Avg Sol':<10} | {'Avg Time (s)':<15} | {'Time/Sol (s)':<15}")
    print("-" * 55)
    for i, mode in enumerate(modes):
        print(f"{name_dct[mode]:<25} | {avg_sol[i]:<10.2f} | {avg_time[i]:<7.4f} ± {std_time[i]:<4.4f} | {avg_time_per_sol[i]:<7.4f}")



def _test_worker(G, mode, T, conn):
    """Worker process that consumes the generator and streams timestamps."""
    try:
        start = time.perf_counter()
        conn.send(0.0)  # Initial timestamp
        
        for _ in G.solve(disp=False, mode=mode):
            elapsed = time.perf_counter() - start
            if elapsed > T:
                break
            conn.send(elapsed)
    except Exception as e:
        conn.send(f"ERROR: {str(e)}")
    finally:
        conn.close()

def run_benchmark(G, modes=[0, 1, 2], nr_tests=10, T=5.0, base_folder="benchmark_results"):
    modes_str = "-".join(map(str, modes))
    folder_name = f"Name={G.name}_T={T}_modes=[{modes_str}]_tests={nr_tests}"
    output_folder = os.path.join(base_folder, folder_name)
    os.makedirs(output_folder, exist_ok=True)

    html_path = os.path.join(output_folder, "puzzle_visualization.html")
    G.write_to_html(html_path)
    
    raw_timestamps = {mode: [] for mode in modes}
    total_tasks = len(modes) * nr_tests
    
    with tqdm(total=total_tasks, desc="Benchmarking", unit="run") as pbar:
        for mode in modes:
            for _ in range(nr_tests):
                parent_conn, child_conn = multiprocessing.Pipe()
                p = multiprocessing.Process(target=_test_worker, args=(G, mode, T, child_conn))
                
                timestamps = []
                p.start()
                child_conn.close() 
                
                start_wait = time.perf_counter()
                while True:
                    # 1. Check if process finished early natively
                    if not p.is_alive() and not parent_conn.poll():
                        break
                        
                    rem_time = T - (time.perf_counter() - start_wait)
                    if rem_time <= 0:
                        break
                        
                    if parent_conn.poll(max(0, rem_time)):
                        try:
                            msg = parent_conn.recv()
                            if isinstance(msg, str) and msg.startswith("ERROR"):
                                print(f"\nWorker error in mode {mode}: {msg}")
                                break
                            timestamps.append(msg)
                        except EOFError:
                            break
                    else:
                        break 
                
                if p.is_alive():
                    p.terminate()
                    p.join()
                
                parent_conn.close()
                
                if not timestamps:
                    timestamps = [0.0]
                    
                raw_timestamps[mode].append(timestamps)
                pbar.update(1)

    payload = {
        "metadata": {"T": T, "modes": modes, "nr_tests": nr_tests},
        "timestamps": raw_timestamps  # Only storing timing metrics
    }
    
    pkl_path = os.path.join(output_folder, "raw_data.pkl")
    with open(pkl_path, "wb") as f:
        pickle.dump(payload, f)
        
    print(f"Timestamps successfully archived to: {pkl_path}")
    return pkl_path


def plot_benchmark(pkl_path, show_trendline=True, fps=12):    
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
        
    metadata = payload["metadata"]
    raw_timestamps = payload["timestamps"]
    T = metadata["T"]
    modes = metadata["modes"]
    output_folder = os.path.dirname(pkl_path)

    # Compute execution time cutoff ceiling (B)
    global_max_time = 0.0
    for mode, runs in raw_timestamps.items():
        for run in runs:
            if run:
                global_max_time = max(global_max_time, run[-1])
                
    B = min(T, global_max_time)
    if B == 0: 
        B = T

    colors = plt.get_cmap('tab10', len(modes))
    time_grid = np.linspace(0, B, 500)
    final_averages = {}
    rate_estimates = {}
    final_max_y = 1
    
    # --- Static Plot ---
    fig_static, ax_static = plt.subplots(figsize=(8, 5))
    
    for i, mode in enumerate(modes):
        c = colors(i)
        runs = raw_timestamps[mode]
        interp_runs = []
        mode_max_time = 0.0
        mode_name = name_dct.get(mode, f"Mode {mode}")
        
        for run in runs:
            run_trimmed = [t for t in run if t <= B]
            counts = np.arange(len(run_trimmed))
            ax_static.step(run_trimmed, counts, where='post', color=c, alpha=0.15)
            final_max_y = max(final_max_y, len(run_trimmed))
            if run_trimmed:
                mode_max_time = max(mode_max_time, run_trimmed[-1])
            
            counts_at_grid = np.searchsorted(run_trimmed, time_grid, side='right') - 1
            interp_runs.append(counts_at_grid)
            
        avg_counts = np.mean(interp_runs, axis=0)
        final_averages[mode] = avg_counts

        rate_estimate = None
        if show_trendline and mode_max_time > 0:
            active_idx = time_grid <= mode_max_time
            if np.sum(active_idx) > 1:
                z = np.polyfit(time_grid[active_idx], avg_counts[active_idx], 1)
                p = np.poly1d(z)
                rate_estimate = z[0]
                # Used label='_nolegend_' to hide trendlines from legend box
                ax_static.plot(time_grid[active_idx], p(time_grid[active_idx]), 
                               color=c, linestyle='--', alpha=0.7, label='_nolegend_')

        if rate_estimate is None:
            rate_estimate = 0.0
        rate_estimates[mode] = rate_estimate

        legend_label = f"{mode_name} ({rate_estimate:.2f} solutions/s)"
        ax_static.plot(time_grid, avg_counts, color=c, alpha=1.0, linewidth=2, label=legend_label)

    ax_static.set_xlim(0, B)
    ax_static.set_ylim(0, final_max_y * 1.1)
    ax_static.set_xlabel("Time (s)")
    ax_static.set_ylabel("Solutions Found")
    ax_static.set_title("Solutions over Time")
    ax_static.legend()
    
    fig_static.savefig(os.path.join(output_folder, "final_trends.png"), bbox_inches='tight')
    plt.close(fig_static)

    # --- Animation ---
    fig_anim, ax_anim = plt.subplots(figsize=(8, 5))
    frames = 60
    
    def update(frame):
        ax_anim.clear()
        current_time = (frame / frames) * B
        current_max_y = 1
        
        for i, mode in enumerate(modes):
            c = colors(i)
            runs = raw_timestamps[mode]
            mode_name = name_dct.get(mode, f"Mode {mode}")
            
            for run in runs:
                run_up_to_t = [t for t in run if t <= current_time]
                if not run_up_to_t: 
                    continue
                counts = np.arange(len(run_up_to_t))
                ax_anim.step(run_up_to_t, counts, where='post', color=c, alpha=0.15)
                current_max_y = max(current_max_y, len(run_up_to_t))
            
            valid_idx = time_grid <= current_time
            if np.any(valid_idx):
                rate_label = f"{mode_name} ({rate_estimates.get(mode, 0.0):.2f} solutions/s)"
                ax_anim.plot(time_grid[valid_idx], final_averages[mode][valid_idx], 
                             color=c, alpha=1.0, linewidth=2, label=rate_label)

        # Dynamic scale bound to 10% of B minimum width
        ax_anim.set_xlim(0, max(current_time, 0.1 * B))
        ax_anim.set_ylim(0, current_max_y * 1.1)
        ax_anim.set_xlabel("Time (s)")
        ax_anim.set_ylabel("Solutions Found")
        ax_anim.set_title(f"Solving Progress (t={current_time:.2f}s / {B:.2f}s)")
        ax_anim.legend(loc="upper left")

    anim = FuncAnimation(fig_anim, update, frames=frames + 1, interval=int(1000/fps))
    anim.save(os.path.join(output_folder, "solving_progress.gif"), writer='pillow', fps=fps)
    plt.close(fig_anim)
    print(f"Visualizations updated in: {output_folder}")