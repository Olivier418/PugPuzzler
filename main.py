import os

from plotting.benchmark import load_benchmark, plot_benchmark, run_benchmark
from classes import SolutionBook, Solution
from plotting.plot_solve_timeline import plot_solve_timeline
from serialization import load_solutionbook, load_solution, load_solution_run, load_game

from plotting.plot_difficultyspace import plot_difficulty_space
import matplotlib.pyplot as plt
import time

from Solver import Solver


if __name__=="__main__":

    IQpuzzler = load_game("games/IQpuzzler")
    solution_book, stats_book = SolutionBook.from_puzzlebook(IQpuzzler.books['main_puzzles'],disp=True)
    solution_book, stats_book = SolutionBook.from_puzzlebook(IQpuzzler.books['pyramid_puzzles'],disp=True)

    IQpuzzlerPRO = load_game("games/IQpuzzlerPRO")
    solution_book, stats_book = SolutionBook.from_puzzlebook(IQpuzzlerPRO.books['main_puzzles'],disp=True)
    solution_book, stats_book = SolutionBook.from_puzzlebook(IQpuzzlerPRO.books['alt_puzzles'],disp=True)
    solution_book, stats_book = SolutionBook.from_puzzlebook(IQpuzzlerPRO.books['pyramid_puzzles'],disp=True)

    # IQquub = load_game("games/IQquub")
    # solution, stats = Solution.from_puzzle(IQquub.puzzles['120'],disp=True)


    # solution_folder = "solutions\\IQpuzzlerPRO\\books\\alt_puzzles\\result_1"
    # solution_book, stats_book = load_solution_run(solution_folder)

    # plot_difficulty_space(solution_book)
    # plot_solve_timeline(solution_book, stats_book)
    # plt.show()  

    # solution_folder = "solutions\\IQpuzzler\\books\\pyramid_puzzles\\result_1"
    # solution_book, stats_book = load_solution_run(solution_folder)

    # plot_difficulty_space(solution_book)
    # plot_solve_timeline(solution_book, stats_book)
    # plt.show()  


    # solution_folder = "solutions\\IQpuzzlerPRO\\books\\main_puzzles\\result_1"
    # solution_book, stats_book = load_solution_run(solution_folder)

    # plot_difficulty_space(solution_book)
    # plot_solve_timeline(solution_book, stats_book)
    # plt.show()  

    # solution_folder = "solutions\\IQpuzzlerPRO\\books\\pyramid_puzzles\\result_1"
    # solution_book, stats_book = load_solution_run(solution_folder)

    # plot_difficulty_space(solution_book)
    # plot_solve_timeline(solution_book, stats_book)
    # plt.show()  
    
    
    
    # EMPTY_MAIN = IQpuzzler.puzzles['empty_main']
    # stats_books, _ = run_benchmark(EMPTY_MAIN, nr_tests = 10, T=10)  # configs default to DEFAULT_CONFIGS
    # stats_books = load_benchmark('benchmarks\\IQpuzzler\\puzzles\\empty_main\\benchmark_1')
    # plot_benchmark(stats_books),
    # plt.show()

    
    # EMPTY_PYR = IQpuzzler.puzzles['empty_pyramid']
    # stats_books, _ = run_benchmark(EMPTY_PYR, nr_tests = 10, T=10)  # configs default to DEFAULT_CONFIGS
    # stats_books = load_benchmark('benchmarks\\IQpuzzler\\puzzles\\empty_pyramid\\benchmark_1')
    # plot_benchmark(stats_books),
    # plt.show()


    # stats_books = load_benchmark('benchmarks\\IQpuzzler\\puzzles\\empty_main\\benchmark_3')
    # plot_benchmark(stats_books)
    # plt.show()


    # solution_folder = "solutions\\IQpuzzlerPRO\\books\\main_puzzles\\result_1"
    # solution_book, stats_book = load_solution_run(solution_folder)

    # plot_difficulty_space(solution_book)
    # plot_solve_timeline(solution_book, stats_book)
    # plt.show()  



    # start = time.perf_counter()
    # list(test_pyramid.solve(disp=True))
    # list(Solver(test_pyramid).solve(disp=True))
    # t1 = time.perf_counter()
    # list(SolverV2(test_pyramid).solve())
    # t2 = time.perf_counter()

    # print(f"old: {t1-start}, new: {t2-t1}")

    # test_pyramid = IQpuzzler.books['pyramid_puzzles']['100']

    # import cProfile, pstats
    # pr = cProfile.Profile()
    # pr.enable()
    # sols_new = {sol.grid.tobytes() for sol in Solver(test_pyramid).solve(seed=0)}
    # pr.disable()
    # pstats.Stats(pr).sort_stats('cumulative').print_stats(20)
