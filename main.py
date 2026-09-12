import os

from classes import SolutionBook
from serialization import save_solutions, load_solutions
from puzzles.IQpuzzler.components import main_puzzle_book, pyramid_puzzle_book
from puzzles.IQpuzzlerPRO.components import alt_puzzle_book, pyramid_puzzle_book as pyramid_puzzle_book_PRO

from benchmark import compare, run_benchmark, plot_benchmark

from plot_difficultyspace import plot_difficulty_space

import matplotlib.pyplot as plt


if __name__=="__main__":

    # solution_book = SolutionBook.from_puzzlebook(main_puzzle_book, disp=True)

    BASE_DIR = os.path.join("solutions","IQpuzzler")
    main_solutions_path = os.path.join(BASE_DIR, "main_solutions.json")
    # save_solutions(solution_book, main_solutions_path)

    board = list(main_puzzle_book.values())[0].board
    blocks = list(main_puzzle_book.values())[0].blocks

    solution_book = load_solutions(main_solutions_path, board, blocks)

    plot_difficulty_space(solution_book)
    plt.show()

    # for puzzle_name, puzzle in pyramid_puzzle_book_PRO.items():
    #     puzzle.print_to_terminal()
    #     solutions = puzzle.solve(disp=False)
    #     print(f"Found {len(list(solutions))} solutions for puzzle '{puzzle_name}'")

        # for solution in solutions:
        #     pass  # or break after the first one, or collect them, etc.