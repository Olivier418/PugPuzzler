import os

from classes import (
    FlatBoard,
    PyramidBoard,
    PuzzleSetup,
)

from classes.puzzle import Puzzle
from serialization import load_block_collection, load_puzzles

BASE_DIR = os.path.join("puzzles","IQpuzzler")


# Load blocks
BLOCKS = load_block_collection(os.path.join(BASE_DIR, "blocks.json"))

# Instantiate boards FIRST before passing into puzzle creation
main_board = FlatBoard(width=11, height=5)
pyramid_board = PyramidBoard(5, 5)

# One PuzzleSetup per board -- computed once here, then shared (via
# PuzzleBook.setup) by every Puzzle in main_puzzle_book/pyramid_puzzle_book,
# and by load_solutions later when loading that book's solved puzzles.
main_setup = PuzzleSetup(BLOCKS, main_board)
pyramid_setup = PuzzleSetup(BLOCKS, pyramid_board)

main_empty = Puzzle(main_setup, name="main_empty")
pyramid_empty = Puzzle(pyramid_setup, name="pyramid_empty")

# Load separate PuzzleBooks cleanly
main_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "main_puzzles.json"), setup=main_setup
)
pyramid_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "pyramid_puzzles.json"), setup=pyramid_setup
)