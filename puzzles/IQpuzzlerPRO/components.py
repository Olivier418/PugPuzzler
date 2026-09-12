from classes import FlatBoard, PyramidBoard
import numpy as np

import os

from classes import (
    FlatBoard,
    PyramidBoard
)

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

# Load separate PuzzleBooks cleanly
# main_puzzle_book = load_puzzles(
#     os.path.join(BASE_DIR, "main_puzzles.json") , board=main_board, blocks=BLOCKS
# )
alt_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "alt_puzzles.json") , board=alt_board, blocks=BLOCKS
)
pyramid_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "pyramid_puzzles.json"), board=pyramid_board, blocks=BLOCKS
)
