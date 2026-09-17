import os

from plotting.benchmark import load_benchmark, plot_benchmark, run_benchmark
from classes import SolutionBook, Solution
from plotting.plot_solve_timeline import plot_solve_timeline
from serialization import load_solutionbook, load_solution, load_solution_run, load_game

from plotting.plot_difficultyspace import plot_difficulty_space
import matplotlib.pyplot as plt
import time

from Solver import Solver
from SolverV2 import SolverV2


if __name__=="__main__":

    IQpuzzler = load_game("games/IQpuzzler")
    solution_book, stats_book = SolutionBook.from_puzzlebook(IQpuzzler.books['pyramid_puzzles'],disp=True)

    IQpuzzlerPRO = load_game("games/IQpuzzlerPRO")
    solution_book, stats_book = SolutionBook.from_puzzlebook(IQpuzzlerPRO.books['main_puzzles'],disp=True)

    IQquub = load_game("games/IQquub")
    solution, stats = Solution.from_puzzle(IQquub.puzzles['120'],disp=True)
    


    solution_folder = "solutions\\IQpuzzlerPRO\\books\\main_puzzles\\result_1"
    solution_book, stats_book = load_solution_run(solution_folder)

    plot_difficulty_space(solution_book)
    plot_solve_timeline(solution_book, stats_book)
    plt.show()  



    # start = time.perf_counter()
    # list(test_pyramid.solve(disp=True))
    # list(Solver(test_pyramid).solve(disp=True))
    # t1 = time.perf_counter()
    # list(SolverV2(test_pyramid).solve())
    # t2 = time.perf_counter()

    # print(f"old: {t1-start}, new: {t2-t1}")

    # import cProfile, pstats
    # pr = cProfile.Profile()
    # pr.enable()
    # sols_new = {sol.grid.tobytes() for sol in Solver(test_pyramid).solve(mode=2, seed=0)}
    # pr.disable()
    # pstats.Stats(pr).sort_stats('cumulative').print_stats(20)


    # BASE_DIR = os.path.join("solutions")

    # solution_book = SolutionBook.from_puzzlebook(pyramid_puzzle_book, disp=True)
    # solution_folder = save_solutions(solution_book, BASE_DIR)

    # # Solutions were already computed and saved in a previous run: reload them 
    # # solution_folder = os.path.join(BASE_DIR, "IQpuzzler_main_puzzles_solutions\\20260914_215811")
    # # solution_book = load_solutionbook(solution_folder)

    # plot_difficulty_space(solution_book)
    # plot_solve_timeline(solution_book)
    # plt.show()

    # BASE_DIR = os.path.join("benchmarks", "IQpuzzlerPRO")

    # books, run_folder = run_benchmark(alt_empty_PRO, modes=[0, 1, 2], nr_tests=30, T=30.0, base_folder=BASE_DIR)
    # books = load_benchmark(run_folder)   # no setup needed anywhere
    # plot_benchmark(books)
    # plt.show()