"""Scratch file of usage examples -- uncomment what you need."""
import matplotlib.pyplot as plt

from benchmark import load_benchmark, run_benchmark
from plotting.plot_benchmark import plot_benchmark
from plotting.plot_puzzlebook import plot_puzzlebook
from plotting.plot_solve_timeline import plot_solve_timeline
from serialization import load_game, load_run, save_puzzlebook
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


    # solution_book, stats_book = solve_puzzlebook(IQpuzzler.books['main_puzzles'], verbose=Verbosity.SUMMARY, save_solutions=True, save_stats=True)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzler.books['pyramid_puzzles'], verbose=Verbosity.SUMMARY, save_solutions=True, save_stats=True)

    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['main_puzzles'], verbose=Verbosity.SUMMARY, save_solutions=True, save_stats=True)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['pyramid_puzzles'], verbose=Verbosity.SUMMARY, save_solutions=True, save_stats=True)
    # solution_book, stats_book = solve_puzzlebook(IQpuzzlerPRO.books['alt_puzzles'], verbose=Verbosity.SUMMARY, save_solutions=True, save_stats=True)

    # solution, stats = solve_puzzle(IQquub.puzzles['89'], verbose=Verbosity.EACH_SOLUTION, save_solutions=True, save_stats=True)
    # solution, stats = solve_puzzle(IQquub.puzzles['90'], verbose=Verbosity.SHOW_SOLUTIONS, save_solutions=True, save_stats=True)
    # solution, stats = solve_puzzle(IQquub.puzzles['120'], verbose=Verbosity.SHOW_SOLUTIONS, save_solutions=True, save_stats=True)



    # puzzles, solution_book, stats_book = load_run("solutions/IQpuzzlerPRO/books/pyramid_puzzles/result_1")
    # plot_puzzlebook(puzzles, stats_book)  # a loaded run carries its own puzzles
    # plot_solve_timeline(puzzles, stats_book)
    # plt.show()

    # trials, folder = run_benchmark(IQpuzzler.puzzles['empty_main'], nr_tests=15, T=15)  # configs default to DEFAULT_CONFIGS
    # trials, folder = run_benchmark(IQpuzzler.puzzles['empty_pyramid'], nr_tests=15, T=15)  # configs default to DEFAULT_CONFIGS
    
    # trials, folder = run_benchmark(IQpuzzlerPRO.puzzles['empty_main'], nr_tests=15, T=15)  # configs default to DEFAULT_CONFIGS
    # trials, folder = run_benchmark(IQpuzzlerPRO.puzzles['empty_pyramid'], nr_tests=15, T=15)  # configs default to DEFAULT_CONFIGS
    # trials, folder = run_benchmark(IQpuzzlerPRO.puzzles['empty_alt'], nr_tests=15, T=15)  # configs default to DEFAULT_CONFIGS
    
    # trials, folder = run_benchmark(IQquub.puzzles['empty_cube'], nr_tests=15, T=15)  # configs default to DEFAULT_CONFIGS


    trials = load_benchmark("benchmarks\\IQpuzzler\\puzzles\\empty_main\\benchmark_1")
    plot_benchmark(trials)
    plt.show()


    trials = load_benchmark("benchmarks\\IQpuzzler\\puzzles\\empty_pyramid\\benchmark_1")
    plot_benchmark(trials)
    plt.show()


    trials = load_benchmark("benchmarks\\IQpuzzlerPRO\\puzzles\\empty_main\\benchmark_1")
    plot_benchmark(trials)
    plt.show()


    trials = load_benchmark("benchmarks\\IQpuzzlerPRO\\puzzles\\empty_pyramid\\benchmark_1")
    plot_benchmark(trials)
    plt.show()



    trials = load_benchmark("benchmarks\\IQpuzzlerPRO\\puzzles\\empty_alt\\benchmark_1")
    plot_benchmark(trials)
    plt.show()





    trials = load_benchmark("benchmarks\\IQquub\\puzzles\\empty_cube\\benchmark_1")
    plot_benchmark(trials)
    plt.show()



    # puzzles, solution_book, stats_book = load_run("solutions/IQpuzzlerPRO/books/pyramid_puzzles/result_1")
    # inhuman_puzzles, inhuman_solutions, inhuman_stats = most_difficult_puzzles(IQpuzzlerPRO.books['pyramid_puzzles']['118'])
    # save_puzzlebook(inhuman_puzzles, IQpuzzlerPRO)  # a home in games/ first: runs refer to their puzzles there
    # solve_puzzlebook(inhuman_puzzles, save_solutions=True, save_stats=True)

    # ax = plot_puzzlebook(puzzles, stats_book, x = "nr_empty_spaces", y = "nr_solutions",z = "time_to_first_solution")
    # plot_puzzlebook(inhuman_puzzles, inhuman_stats, ax=ax)  # overlay: axes come from ax

    # plt.show()
