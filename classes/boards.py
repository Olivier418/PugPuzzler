import numpy as np

from .lattice import Lattice


class Board:
    def __init__(self, cells: np.ndarray, offset_adjacency: np.ndarray):
        self.cells = np.asarray(cells, dtype=bool)
        self.lattice = Lattice(offset_adjacency)
        if self.lattice.ndim != self.cells.ndim:
            raise ValueError(
                f"offset_adjacency describes {self.lattice.ndim} axes, but cells has "
                f"{self.cells.ndim} dimensions; they must match."
            )
        self.ndim = self.cells.ndim
        self.shape = self.cells.shape


class FlatBoard(Board):
    def __init__(self, cells: np.ndarray = None, width: int = None, height: int = None):
        if cells is not None:
            cells = np.asarray(cells)
        elif (width is not None) and (height is not None):
            cells = np.ones((width, height), dtype=bool)
        else:
            raise ValueError("Either cells or both width and height must be provided.")
        # Plain rectangular lattice: neither axis is offset from the other.
        offset_adjacency = np.zeros((2, 2), dtype=bool)
        super().__init__(cells, offset_adjacency)


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