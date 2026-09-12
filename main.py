import os

from classes import SolutionBook
from serialization import save_solutions, load_solutions
from puzzles.IQpuzzler.components import main_puzzle_book, pyramid_puzzle_book
from puzzles.IQpuzzlerPRO.components import alt_puzzle_book, pyramid_puzzle_book as pyramid_puzzle_book_PRO

from plot_difficultyspace import plot_difficulty_space
import matplotlib.pyplot as plt


if __name__=="__main__":
    BASE_DIR = os.path.join("solutions","IQpuzzlerPRO")
    solutions_path = os.path.join(BASE_DIR, "alt_solutions.json")

    # solution_book = SolutionBook.from_puzzlebook(alt_puzzle_book, disp=True)
    # save_solutions(solution_book, solutions_path)

    # Solutions were already computed and saved in a previous run: reload
    # them using the setup main_puzzle_book already has, rather than
    # re-running the expensive placement search a second time.
    solution_book = load_solutions(solutions_path, alt_puzzle_book.setup)

    plot_difficulty_space(solution_book)
    plt.show()

    # for puzzle_name, puzzle in pyramid_puzzle_book_PRO.items():
    #     puzzle.print_to_terminal()
    #     solutions = puzzle.solve(disp=False)
    #     print(f"Found {len(list(solutions))} solutions for puzzle '{puzzle_name}'")

        # for solution in solutions:
        #     pass  # or break after the first one, or collect them, etc.