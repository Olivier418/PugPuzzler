import os

from classes import SolutionBook
from serialization import save_solutions, load_solutions
from puzzles.IQpuzzler.components import main_puzzle_book, pyramid_puzzle_book
from puzzles.IQpuzzlerPRO.components import (main_puzzle_book as main_puzzle_book_PRO,
                                            alt_puzzle_book as alt_puzzle_book_PRO, 
                                            pyramid_puzzle_book as pyramid_puzzle_book_PRO)

from plot_difficultyspace import plot_difficulty_space
import matplotlib.pyplot as plt


if __name__=="__main__":
    BASE_DIR = os.path.join("solutions","IQpuzzler")
    solutions_path = os.path.join(BASE_DIR, "pyramid_solutions.json")

    solution_book = SolutionBook.from_puzzlebook(pyramid_puzzle_book, disp=True)
    save_solutions(solution_book, solutions_path)

    # Solutions were already computed and saved in a previous run: reload
    # them using the setup main_puzzle_book already has, rather than
    # re-running the expensive placement search a second time.
    # solution_book = load_solutions(solutions_path, pyramid_puzzle_book_PRO.setup)

    plot_difficulty_space(solution_book)
    plt.show()

    # for solved_puzzle_name, solved_puzzle in solution_book.items():
    #     solved_puzzle.puzzle.print_to_terminal()
    #     for i, solution in enumerate(solved_puzzle.solutions):
    #         print(f"Solution {i+1}:")
    #         solution.print_to_terminal()

