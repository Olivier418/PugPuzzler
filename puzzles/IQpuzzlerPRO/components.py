from classes import FlatBoard, PyramidBoard
import numpy as np

import os

from classes import (
    FlatBoard,
    PyramidBoard
)

from classes.puzzle import Puzzle, PuzzleSetup
from serialization import load_block_collection, load_puzzles

BASE_DIR = os.path.join("puzzles","IQpuzzlerPRO")


# Load blocks
BLOCKS = load_block_collection(os.path.join(BASE_DIR, "blocks.json"))

# Instantiate boards FIRST before passing into puzzle creation
main_board = FlatBoard(width=11, height=5)
alt_board = FlatBoard(cells=np.array([  [0, 1, 1, 1, 1, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 0, 0, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 0, 0],
                                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                                        [0, 0, 1, 1, 1, 1, 1, 1, 1],
                                        [0, 0, 1, 1 ,1 ,1 ,1 ,1, 1],
                                        [0, 0, 0, 0, 1, 1, 1, 1, 1],
                                        [0, 0, 0, 0, 1, 1, 1, 1, 0]], dtype=bool))
pyramid_board = PyramidBoard(5,5)

main_setup = PuzzleSetup(BLOCKS, main_board)
pyramid_setup = PuzzleSetup(BLOCKS, pyramid_board)
alt_setup = PuzzleSetup(BLOCKS, alt_board)

main_empty = Puzzle(main_setup, name="main_empty")
pyramid_empty = Puzzle(pyramid_setup, name="pyramid_empty")
alt_empty = Puzzle(alt_setup, name="alt_empty")

# Load separate PuzzleBooks cleanly
main_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "main_puzzles.json"), setup=main_setup
)
alt_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "alt_puzzles.json"), setup=alt_setup
)
pyramid_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "pyramid_puzzles.json"), setup=pyramid_setup
)
