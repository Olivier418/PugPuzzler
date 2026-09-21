"""Scratch file of usage examples -- uncomment what you need."""
import matplotlib.pyplot as plt

from benchmark import load_benchmark, run_benchmark
from plotting.plot_benchmark import plot_benchmark
from plotting.plot_difficultyspace import plot_difficulty_space
from plotting.plot_solve_timeline import plot_solve_timeline
from serialization import load_game, load_solution_run
from solving import Verbosity, solve_puzzle, solve_puzzlebook


if __name__ == "__main__":
    IQpuzzler = load_game("games/IQpuzzler")
    IQpuzzlerPRO = load_game("games/IQpuzzlerPRO")
    IQquub = load_game("games/IQquub")

    # solution_book, stats_book = solve_puzzlebook(IQpuzzler.books['main_puzzles'], verbose=Verbosity.SUMMARY)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzler.books['pyramid_puzzles'], verbose=Verbosity.SUMMARY)

    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['main_puzzles'], verbose=Verbosity.SUMMARY)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['pyramid_puzzles'], verbose=Verbosity.SUMMARY)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['alt_puzzles'], verbose=Verbosity.SUMMARY)

    solution, stats = solve_puzzle(IQpuzzler.puzzles['empty_pyramid'], verbose=Verbosity.EACH_SOLUTION,up_to_symmetry=False)
    # solution, stats = solve_puzzle(IQquub.puzzles['90'], verbose=Verbosity.SHOW_SOLUTIONS)
    # solution, stats = solve_puzzle(IQquub.puzzles['120'], verbose=Verbosity.SHOW_SOLUTIONS)

    # solution_book, stats_book = load_solution_run("solutions/IQpuzzlerPRO/books/pyramid_puzzles/result_1")
    # plot_difficulty_space(solution_book)
    # plot_solve_timeline(solution_book, stats_book)
    # plt.show()

    # stats_books, folder = run_benchmark(IQpuzzler.puzzles['empty_main'], nr_tests=10, T=10)  # configs default to DEFAULT_CONFIGS
    # stats_books = load_benchmark("benchmarks\\IQpuzzler\\puzzles\\empty_pyramid\\benchmark_1")
    # plot_benchmark(stats_books, T=10)
    # plt.show()

    # stats_books, folder = run_benchmark(IQpuzzler.puzzles['empty_pyramid'], nr_tests=10, T=10)  # configs default to DEFAULT_CONFIGS
        
    # stats_books, folder = run_benchmark(IQpuzzler.books['main_puzzles']['65'], nr_tests=10, T=10)  # configs default to DEFAULT_CONFIGS
        
