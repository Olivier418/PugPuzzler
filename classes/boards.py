import itertools
from functools import cached_property

import numpy as np


class Board:
    """A board's cells (bool array, True = a cell) and the geometry of its
    axes: `offset_adjacency[i, j]` says whether axes i and j are offset by
    half a cell from each other (e.g. a pyramid's layers) rather than
    orthogonal. Every offset pattern on at most 3 axes is a valid lattice;
    beyond 3 some aren't, which nothing here checks."""

    def __init__(self, cells: np.ndarray, offset_adjacency: np.ndarray):
        self.cells = np.asarray(cells, dtype=bool)
        self.offset_adjacency = np.asarray(offset_adjacency, dtype=bool)

    @cached_property
    def gram2(self) -> np.ndarray:
        """Twice the Gram matrix of the axes' unit vectors, in integers: 2 on
        the diagonal, 1 between offset axes, 0 between orthogonal ones."""
        return 2 * np.eye(self.cells.ndim, dtype=int) + self.offset_adjacency.astype(int)

    @cached_property
    def unit_vectors(self) -> list[np.ndarray]:
        """Every vector v in {-1, 0, 1}^ndim of unit length (v @ gram2 @ v ==
        2): every direction a block's cells can step in on this board."""
        vectors = (np.array(v) for v in itertools.product((-1, 0, 1), repeat=self.cells.ndim))
        return [v for v in vectors if v @ self.gram2 @ v == 2]


class RegularBoard(Board):
    """An orthogonal board: the given `cells`, or a full width x depth
    (x height) box."""

    def __init__(self, cells: np.ndarray = None, width: int = None, depth: int = None, height: int = None):
        if cells is None:
            cells = np.ones((width, depth) if height is None else (width, depth, height), dtype=bool)
        cells = np.asarray(cells, dtype=bool)
        super().__init__(cells, np.zeros((cells.ndim, cells.ndim), dtype=bool))


class PyramidBoard(Board):
    def __init__(self, width: int, depth: int):
        height = min(width, depth)
        shape = (width, depth, height)
        i_idx, j_idx, h_idx = np.indices(shape)
        cells = (h_idx < height - i_idx) & (h_idx < height - j_idx)

        # Width and depth axes are orthogonal to each other, but each is
        # offset by 1/2 from the height axis (successive layers are
        # shifted diagonally by half a cell).
        offset_adjacency = np.array([
            [False, False, True],
            [False, False, True],
            [True,  True,  False],
        ])

        super().__init__(cells, offset_adjacency)
