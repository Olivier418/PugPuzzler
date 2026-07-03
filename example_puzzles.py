from default_settings import WIDTH, HEIGHT, BLOCKS
from classes import FlatGame
import numpy as np

P0grid = [
    [ 2,  2,  2, -1, -1, -1, -1,  5,  5,  1,  1],
    [ 2, -1,  2, -1, -1, -1, -1,  6,  5,  5,  1],
    [-1, -1,  7, -1, -1, -1,  6,  6,  6,  5,  1],
    [ 8,  8,  7, -1, -1, -1, -1,  6,  0,  0,  0],
    [ 8,  8,  7,  7,  7, -1, -1, -1, -1,  0,  0],
]
P0 = FlatGame(WIDTH, HEIGHT, BLOCKS, name="P0", grid=np.array(P0grid, dtype=int).T)

P1grid = [
    [ 2,  2,  2, -1, -1, -1, -1,  5,  5,  1,  1],
    [ 2, -1,  2, -1, -1, -1, -1,  6,  5,  5,  1],
    [-1, -1, -1, -1, -1, -1,  6,  6,  6,  5,  1],
    [-1, -1, -1, -1, -1, -1, -1,  6,  0,  0,  0],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
]
P1 = FlatGame(WIDTH, HEIGHT, BLOCKS, name="P1", grid = np.array(P1grid, dtype=int).T)


P2grid = [
    [-1, -1, -1, -1, -1, -1, -1,  5,  5,  1,  1],
    [-1, -1, -1, -1, -1, -1, -1,  6,  5,  5,  1],
    [-1, -1, -1, -1, -1, -1,  6,  6,  6,  5,  1],
    [-1, -1, -1, -1, -1, -1, -1,  6,  0,  0,  0],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1,  0,  0],
]
P2 = FlatGame(WIDTH, HEIGHT, BLOCKS, name="P2", grid = np.array(P2grid, dtype=int).T)

P71grid = [
    [ 9,  9,  9,  9,  1,  1,  1,  3,  3,  3,  3],
    [-1,  9, -1, -1,  1, -1, -1,  3, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
]
P71 = FlatGame(WIDTH, HEIGHT, BLOCKS, name="P71", grid = np.array(P71grid, dtype=int).T)

P72grid = [
    ['A', 'A', 'A', 'J', 'J', 'J', 'J', 'D', 'D', 'D', 'D'],
    [' ', ' ', 'A', ' ', ' ', ' ', ' ', ' ', ' ', 'D', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
    [' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' '],
]
P72 = FlatGame(WIDTH, HEIGHT, BLOCKS, name="P72", grid = np.array(P72grid, dtype=str).T)

