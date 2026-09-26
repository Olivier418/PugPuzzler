"""Scratch file of usage examples -- uncomment what you need."""
import matplotlib.pyplot as plt

from benchmark import load_benchmark, run_benchmark
from plotting.plot_benchmark import plot_benchmark
from plotting.plot_bounds import plot_bounds
from plotting.plot_puzzle_stats import plot_puzzle_stats
from plotting.plot_solve_timeline import plot_solve_timeline
from serialization import (
    latest_complete_run, load_bounds, load_game, load_puzzle_run, load_run, save_bounds, save_puzzlebook,
)
from solving import Verbosity, empty_puzzle, solve_puzzle, solve_puzzlebook

from puzzle_bounds import compute_puzzle_bounds
from placements import print_placement_table


if __name__ == "__main__":
    IQpuzzler = load_game("games/IQpuzzler")
    IQpuzzlerPRO = load_game("games/IQpuzzlerPRO")
    IQquub = load_game("games/IQquub")

    # print_placement_table(IQpuzzler)
    # print_placement_table(IQpuzzlerPRO)
    # print_placement_table(IQquub)



    # ....first
    empty = empty_puzzle(IQpuzzlerPRO, "alt")
    empty_solution, empty_stats = load_puzzle_run(latest_complete_run(empty), empty)

    (lower_puzzles, lower_stats), (upper_puzzles, upper_stats) = compute_puzzle_bounds(empty_solution, empty_stats)
    save_bounds(IQpuzzlerPRO, (lower_puzzles, lower_stats), (upper_puzzles, upper_stats))

    # ...and from then on:
    (lower_puzzles, lower_stats), (upper_puzzles, upper_stats) = load_bounds(IQpuzzlerPRO, "alt")
    ax = plot_bounds(lower_puzzles, lower_stats, upper_puzzles, upper_stats)

    puzzles, solution_book, stats_book = load_run("solutions/IQpuzzlerPRO/books/alt_puzzles/result_1")
    plot_puzzle_stats(puzzles, stats_book,ax=ax)  # a loaded run carries its own puzzles
    plt.show()



    # puzzles, solution_book, stats_book = load_run("solutions/IQpuzzlerPRO/books/pyramid_puzzles/result_1")
    # plot_puzzle_stats(puzzles, stats_book)  # a loaded run carries its own puzzles
    # plot_solve_timeline(puzzles, stats_book)
    # plt.show()


    # trials = load_benchmark("benchmarks\\IQpuzzler\\puzzles\\empty_main\\benchmark_1")
    # plot_benchmark(trials)
    # plt.show()
