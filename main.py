"""Scratch file of usage examples -- uncomment what you need."""
import matplotlib.pyplot as plt

from benchmark import load_benchmark, run_benchmark
from plotting.plot_benchmark import plot_benchmark
from plotting.plot_puzzlebook import plot_puzzlebook
from plotting.plot_solve_timeline import plot_solve_timeline
from serialization import load_game, load_solution_run, save_puzzlebook
from solving import Verbosity, solve_puzzle, solve_puzzlebook

from most_difficult_puzzle import most_difficult_puzzles
from placements import print_placement_table


if __name__ == "__main__":
    IQpuzzler = load_game("games/IQpuzzler")
    IQpuzzlerPRO = load_game("games/IQpuzzlerPRO")
    IQquub = load_game("games/IQquub")

    # print_placement_table(IQpuzzler)
    # print_placement_table(IQpuzzlerPRO)
    # print_placement_table(IQquub)


    # solution_book, stats_book = solve_puzzlebook(IQpuzzler.books['main_puzzles'], verbose=Verbosity.SUMMARY, save=True)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzler.books['pyramid_puzzles'], verbose=Verbosity.SUMMARY, save=True)

    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['main_puzzles'], verbose=Verbosity.SUMMARY, save=True)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['pyramid_puzzles'], verbose=Verbosity.SUMMARY, save=True)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['alt_puzzles'], verbose=Verbosity.SUMMARY, save=True)

    # solution, stats = solve_puzzle(IQquub.puzzles['89'], verbose=Verbosity.EACH_SOLUTION, save=True)
    # solution, stats = solve_puzzle(IQquub.puzzles['90'], verbose=Verbosity.SHOW_SOLUTIONS, save=True)
    # solution, stats = solve_puzzle(IQquub.puzzles['120'], verbose=Verbosity.SHOW_SOLUTIONS, save=True)



    # solution_book, stats_book = load_solution_run("solutions/IQpuzzlerPRO/books/pyramid_puzzles/result_1")
    # plot_puzzlebook(solution_book)  # a loaded run carries its own puzzles
    # plot_solve_timeline(solution_book, stats_book)
    # plt.show()

    # stats_books, folder = run_benchmark(IQpuzzler.puzzles['empty_main'], nr_tests=10, T=10)  # configs default to DEFAULT_CONFIGS
    # stats_books = load_benchmark("benchmarks\\IQpuzzler\\puzzles\\empty_main\\benchmark_3")
    # plot_benchmark(stats_books)
    # plt.show()



    # solution_book, stats_book = load_solution_run("solutions/IQpuzzlerPRO/books/pyramid_puzzles/result_1")
    # inhuman_puzzles, inhuman_solutions, inhuman_stats = most_difficult_puzzles(IQpuzzlerPRO.books['pyramid_puzzles']['118'])
    # save_puzzlebook(inhuman_puzzles, IQpuzzlerPRO)  # a home in games/ first: runs refer to their puzzles there
    # solve_puzzlebook(inhuman_puzzles, save=True)

    # ax = plot_puzzlebook(solution_book, stats_book, x = "nr_empty_spaces", y = "nr_solutions",z = "time_to_first_solution")
    # plot_puzzlebook(inhuman_solutions, inhuman_stats, ax=ax)  # overlay: axes come from ax

    # plt.show()
