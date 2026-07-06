from default_settings import WIDTH, HEIGHT, BLOCKS, EMPTY
from example_puzzles import P72
from classes import FlatGame, Block, Game
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt
import time
import math


def most_difficult_puzzle(initial_game: Game = P72):
    """Return the most difficult puzzle in the default settings.
    Puzzle difficulty is defined as nr_unfilled_positions/nr_solutions"""
    def find_better_game(current_game: Game, current_difficulty: float):
        empty_candidates = np.arange(
            int(np.floor(current_difficulty)) + 1,
            np.prod(current_game.grid.shape) + 1
        )
        thresholds = np.ceil(empty_candidates / current_difficulty).astype(int) - 1

        for empty_count, threshold in zip(empty_candidates, thresholds):

            ...

            for subset in ...:
                ...

                for combination in ...:
                    G = ...

                    nr_solutions = 0
                    for nr_solutions, _ in enumerate(G.solve(disp=False, mode=2), start=1):
                        if nr_solutions > threshold:
                            break
                    else:
                        # Executed only if the solver was exhausted.
                        difficulty = empty_count / nr_solutions

                        return G, difficulty, empty_count, nr_solutions

        return None, None, None, None

    current_game = initial_game

    nr_solutions = len(list(current_game.solve(disp=False, mode=2)))
    nr_empty = np.sum(current_game.grid == -1)
    current_difficulty = nr_empty / nr_solutions

    record_games = [current_game]
    record_difficulties = [current_difficulty]
    record_empty = [nr_empty]
    record_nr_sols = [nr_solutions]

    while True:
        current_game, current_difficulty, current_empty, current_nr_sol = find_better_game(current_game, current_difficulty)

        if current_game is None:
            print("No harder puzzle found.")
            break


        record_games.append(current_game)
        record_difficulties.append(current_difficulty)
        record_empty.append(current_empty)
        record_nr_sols.append(current_nr_sol)

        print(f"Found game with difficulty {current_difficulty:.3f}")
    
    return current_game



# def plot_difficulty(points, best_score=None, savepath="most_difficult_scatter.png"):
#     if points.size == 0:
#         print("No points to plot.")
#         return

#     xs = points[:, 0]  # nr_unfilled
#     ys = points[:, 1]  # nr_solutions

#     plt.figure(figsize=(8, 6))
#     plt.scatter(xs, ys, c='C0', alpha=0.6)
#     plt.xlabel("Nr unfilled positions")
#     plt.ylabel("Nr solutions")
#     plt.title("Tested puzzles: solutions vs unfilled positions")
#     plt.yscale('log')
#     if best_score is not None:
#         plt.annotate(f"best={best_score:.3f}", xy=(0.02, 0.95), xycoords='axes fraction')
#     plt.grid(True, which='both', ls='--', alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(savepath)
#     print(f"Plot saved to: {savepath}")

# plot_difficulty(points, best)


if __name__ == '__main__':
    most_difficult_puzzle()