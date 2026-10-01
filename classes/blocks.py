from collections import UserDict

import numpy as np

from ._utils import _assert_unique


class Block:
    def __init__(self, positions, letter, terminal_color):
        # positions: a bool mask of the block's cells, on its own orthogonal grid
        self.coords = np.argwhere(np.asarray(positions, dtype=bool))
        self.ndim = self.coords.shape[1]
        self.terminal_color = terminal_color
        self.letter = letter
        self.count = len(self.coords)


class BlockCollection(UserDict):
    """The blocks, keyed 0..n-1 in the order given: a block's key is also
    its column in a solution row and its position in the kernel."""

    def __init__(self, *blocks):
        _assert_unique(blocks, lambda b: b.letter, "block letters")
        super().__init__(enumerate(blocks))
