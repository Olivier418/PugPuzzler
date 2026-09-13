import os

from benchmark import load_benchmark, plot_benchmark, run_benchmark
from classes import SolutionBook
from plot_solve_timeline import plot_solve_timeline
from serialization import save_solutions, load_solutions
from puzzles.IQpuzzler.components import (main_puzzle_book, main_empty,
                                          pyramid_puzzle_book, pyramid_empty)
from puzzles.IQpuzzlerPRO.components import (main_puzzle_book as main_puzzle_book_PRO, main_empty as main_empty_PRO,
                                            alt_puzzle_book as alt_puzzle_book_PRO, alt_empty as alt_empty_PRO,
                                            pyramid_puzzle_book as pyramid_puzzle_book_PRO, pyramid_empty as pyramid_empty_PRO)

from plot_difficultyspace import plot_difficulty_space
import matplotlib.pyplot as plt


if __name__=="__main__":
    # BASE_DIR = os.path.join("solutions","IQpuzzlerPRO")
    # solutions_path = os.path.join(BASE_DIR, "pyramid_solutions.json")

    # solution_book = SolutionBook.from_puzzlebook(pyramid_puzzle_book_PRO, disp=True)
    # save_solutions(solution_book, solutions_path)

    # # Solutions were already computed and saved in a previous run: reload them 
    # # solution_book = load_solutions(solutions_path, pyramid_puzzle_book.setup)

    # # plot_difficulty_space(solution_book)
    # plot_solve_timeline(solution_book)
    # plt.show()

    BASE_DIR = os.path.join("benchmarks", "IQpuzzlerPRO")

    books = run_benchmark(main_empty_PRO, modes=[0, 1, 2], nr_tests=35, T=20.0, base_folder=BASE_DIR)
    # writes BASE_DIR/<some_puzzle.name>/mode0_test0.json ... mode2_test9.json

    # books = load_benchmark(main_empty_PRO.name, main_empty_PRO.setup, base_folder=BASE_DIR)

    plot_benchmark(books)
    plt.show()