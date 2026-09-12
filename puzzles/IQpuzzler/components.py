import os

from classes import (
    FlatBoard,
    PyramidBoard
)

from serialization import load_block_collection, load_puzzles

BASE_DIR = os.path.join("puzzles","IQpuzzler")


# Load blocks
BLOCKS = load_block_collection(os.path.join(BASE_DIR, "blocks.json"))

# Instantiate boards FIRST before passing into puzzle creation
main_board = FlatBoard(width=11, height=5)
pyramid_board = PyramidBoard(5, 5)

# Load separate PuzzleBooks cleanly
main_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "main_puzzles.json") , board=main_board, blocks=BLOCKS
)
pyramid_puzzle_book = load_puzzles(
    os.path.join(BASE_DIR, "pyramid_puzzles.json"), board=pyramid_board, blocks=BLOCKS
)
