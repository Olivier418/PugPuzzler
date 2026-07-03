from default_settings import WIDTH, HEIGHT, BLOCKS, EMPTY
from classes import FlatGame, Block
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import time
import math

def most_difficult_puzzle(initial_game=None):
    """Return the most difficult puzzle in the default settings.
    Puzzle difficulty is defined as nr_unfilled_positions/nr_solutions"""
    # For every piece, place it in every possible position and orientation, and count the number of solutions.
    # The puzzle with the fewest solutions per unfilled position is the most difficult.
    

    # --- computation separated from plotting ---
    def compute_tested_points(initial_game=None, initial_record=None, max_placed=3, max_evals=5000):
        """
        Explore candidate puzzles by placing up to `max_placed` additional pieces.
        - `initial_game`: a `Game` instance that sets the starting threshold (optional).
        Returns (points_array, best_score) where points_array is Nx2 of (nr_unfilled, nr_solutions).
        """
        placements = EMPTY.solver.placements
        total_cells = WIDTH * HEIGHT

        results = []  # tuples (nr_unfilled, nr_solutions)

        # derive initial best score
        if initial_game is not None:
            bu = int(np.sum(initial_game.grid == -1))
            try:
                bs = sum(1 for _ in initial_game.solve(mode=2))
            except Exception:
                bs = 1
            best_score = float(bu) / float(bs)
        else:
            # no initial record: start from the empty puzzle (difficulty 0)
            best_score = 0.0

        eval_count = 0
        grid = EMPTY.grid.copy()
        start_time = time.perf_counter()

        # Precompute block counts list
        blocks = list(BLOCKS.items())  # (idx, Block)
        counts = [b.count for _, b in blocks]

        # Find all subsets of block indices whose total filled cells == required_filled
        def subsets_for_filled(required_filled):
            res = []
            def backtrack(i, cur_sum, cur_set):
                if cur_sum == required_filled:
                    res.append(list(cur_set))
                    return
                if cur_sum > required_filled or i >= len(blocks):
                    return
                idx_i, block_i = blocks[i]
                cur_set.append(idx_i)
                backtrack(i + 1, cur_sum + block_i.count, cur_set)
                cur_set.pop()
                backtrack(i + 1, cur_sum, cur_set)
            backtrack(0, 0, [])
            return res

        # For a given subset of block indices, iterate over non-overlapping placements
        def test_subset(subset):
            nonlocal eval_count, best_score
            subset_sorted = sorted(subset, key=lambda k: BLOCKS[k].count, reverse=True)

            def rec_place(i):
                nonlocal eval_count, best_score
                if i >= len(subset_sorted):
                    u = int(np.sum(grid == -1))
                    allowed_max = None
                    if best_score > 0:
                        allowed_max = int(math.floor(u / best_score))
                    cnt = 0
                    try:
                        for _ in FlatGame(WIDTH, HEIGHT, BLOCKS, name="CAND", grid=grid.reshape((WIDTH, HEIGHT))).solve(mode=2):
                            cnt += 1
                            if allowed_max is not None and cnt > allowed_max:
                                break
                    except Exception:
                        cnt = 0
                    if cnt == 0:
                        return False
                    difficulty = float(u) / cnt
                    print(f"Testing candidate: unfilled={u}, solutions={cnt}, difficulty={difficulty:.4f}")
                    results.append((u, cnt))
                    eval_count += 1
                    if cnt > 0 and difficulty > best_score:
                        best_score = difficulty
                        print(f"New best score: {best_score:.4f} (unfilled={u}, solutions={cnt})")
                        return True
                    if eval_count >= max_evals:
                        return True
                    return False
                b_idx = subset_sorted[i]
                for placement in placements[b_idx]:
                    if np.any(grid.flat[placement] != -1):
                        continue
                    grid.flat[placement] = b_idx
                    stop = rec_place(i + 1)
                    grid.flat[placement] = -1
                    if stop:
                        return True
                return False

            return rec_place(0)

        # iterate target unfilled counts in increasing order (smallest improvement first)
        start_u = int(math.floor(best_score)) + 1 if best_score > 0 else 1
        total_available = sum(counts)
        for target_u in range(start_u, total_cells + 1):
            required_filled = total_cells - target_u
            if required_filled < 0 or required_filled > total_available:
                continue
            subsets = subsets_for_filled(required_filled)
            if not subsets:
                continue
            for subset in subsets:
                stop = test_subset(subset)
                if stop:
                    elapsed = time.perf_counter() - start_time
                    print(f"Evaluated {eval_count} puzzles in {elapsed:.2f}s; best_score={best_score:.4f}")
                    return np.array(results, dtype=int), best_score

        elapsed = time.perf_counter() - start_time
        print(f"Evaluated {eval_count} puzzles in {elapsed:.2f}s; best_score={best_score:.4f}")
        return np.array(results, dtype=int), best_score

    def plot_difficulty(points, best_score=None, savepath="most_difficult_scatter.png"):
        if points.size == 0:
            print("No points to plot.")
            return

        xs = points[:, 0]  # nr_unfilled
        ys = points[:, 1]  # nr_solutions

        plt.figure(figsize=(8, 6))
        plt.scatter(xs, ys, c='C0', alpha=0.6)
        plt.xlabel("Nr unfilled positions")
        plt.ylabel("Nr solutions")
        plt.title("Tested puzzles: solutions vs unfilled positions")
        plt.yscale('log')
        if best_score is not None:
            plt.annotate(f"best={best_score:.3f}", xy=(0.02, 0.95), xycoords='axes fraction')
        plt.grid(True, which='both', ls='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(savepath)
        print(f"Plot saved to: {savepath}")

    points, best = compute_tested_points()
    plot_difficulty(points, best)


if __name__ == '__main__':
    most_difficult_puzzle()