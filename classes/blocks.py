from collections import UserDict
from typing import Optional

import numpy as np

from ._utils import _assert_unique
from .lattice import Lattice


class Block:
    def __init__(self, positions, rgb, letter, terminal_color, offset_adjacency: Optional[np.ndarray] = None):
        positions = np.asarray(positions)
        self.coords = np.argwhere(positions) if positions.dtype == bool else positions
        self.ndim = self.coords.shape[1]
        self.rgb = rgb
        self.terminal_color = terminal_color
        self.letter = letter
        self.count = len(self.coords)

        # Default: standard square/cubic block, no axes offset from one another.
        if offset_adjacency is None:
            offset_adjacency = np.zeros((self.ndim, self.ndim), dtype=bool)
        self.lattice = Lattice(offset_adjacency)
        if self.lattice.ndim != self.ndim:
            raise ValueError(
                f"offset_adjacency describes {self.lattice.ndim} axes, but the block's "
                f"coordinates have {self.ndim} dimensions; they must match."
            )


class BlockCollection(UserDict):
    def __init__(self, *blocks):
        _assert_unique(blocks, lambda b: b.letter, "block letters")
        super().__init__(enumerate(blocks))