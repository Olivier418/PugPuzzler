import copy
from collections import UserDict
import itertools

import numpy as np
from colorama import Style

from constants import EMPTY, OUTSIDE_BOARD, UNPLACED
from Solver import Solver

from ._utils import _assert_unique, _shared
from .blocks import Block, BlockCollection
from .boards import Board


class PuzzleSetup:
    """The fixed setup shared by every puzzle in a book: a board, a block
    collection, and the (expensive to compute) valid placements for each
    block on that board.

    None of this depends on which specific solution a puzzle uses, so it
    is computed once here and then shared BY REFERENCE across every
    Game/Puzzle built on top of it, instead of being recomputed from
    scratch for each one.
    """

    def __init__(self, blocks: BlockCollection, board: Board):
        self.board = board
        self.blocks = blocks
        self.placements = self._compute_placement_indices()
        self._validate()

    def _valid_orientations(self, block: Block) -> list[np.ndarray]:
        k = block.ndim

        G_board2 = self.board.lattice.gram2
        G_piece2 = block.lattice.gram2

        # 1. All unit vectors in board space, i.e. v^T @ (2G) @ v == 2.
        # Cached on the board's lattice, so this is computed once per board
        # and reused across every block, rather than recomputed each time.
        candidate_vecs = self.board.lattice.unit_vectors

        # 2. Find valid transformation matrices M composed of orthogonal unit vectors
        unique_shapes = []
        seen = set()

        for cols in itertools.product(candidate_vecs, repeat=k):
            M = np.column_stack(cols)  # Shape (d, k)

            # Check if M preserves physical distances in pure integer arithmetic:
            # M^T @ (2G_board) @ M == 2G_piece
            if np.array_equal(M.T @ G_board2 @ M, G_piece2):
                board_coords = block.coords @ M.T
                board_coords -= board_coords.min(axis=0)

                canon = board_coords[np.lexsort(board_coords.T[::-1])]
                key = canon.tobytes()
                if key not in seen:
                    seen.add(key)
                    unique_shapes.append(canon)

        return unique_shapes

    def _compute_placement_indices(self):
        result = {}
        cells = self.board.cells
        board_shape = self.board.shape
        ndim = self.board.ndim

        for block_idx, block in self.blocks.items():
            shapes = self._valid_orientations(block)
            result_chunks = []

            for cell_coords in shapes:
                extent = cell_coords.max(axis=0) + 1
                n_starts = np.array(board_shape) - extent + 1
                if np.any(n_starts <= 0):
                    continue

                ranges = [np.arange(n) for n in n_starts]
                starts = np.stack(np.meshgrid(*ranges, indexing='ij'), axis=-1).reshape(-1, ndim)

                placed = starts[:, None, :] + cell_coords[None, :, :]
                idx_tuple = tuple(placed[..., d] for d in range(ndim))

                valid_mask = cells[idx_tuple].all(axis=1)
                valid_placed = placed[valid_mask]

                if valid_placed.size > 0:
                    flat_idx_tuple = tuple(valid_placed[..., d] for d in range(ndim))
                    flat = np.ravel_multi_index(flat_idx_tuple, board_shape)
                    result_chunks.append(flat)

            result[block_idx] = (
                np.concatenate(result_chunks, axis=0)
                if result_chunks
                else np.empty((0, block.count), dtype=np.int64)
            )
        return result

    def _validate(self):
        total_cells = sum(b.count for b in self.blocks.values())
        if total_cells != np.sum(self.board.cells):
            raise ValueError("Total block cells do not match board cells.")

        for idx, placements in self.placements.items():
            if len(placements) == 0:
                raise ValueError(f"Block {idx} has no valid placements on the board.")

    def print_to_terminal(self, grid: np.ndarray, print_letters: bool = True):
        """Render any grid shaped like this setup's board -- a Game's own
        grid, a solved grid from Solver, or otherwise. This only depends
        on board/blocks, both owned by the setup, not on any particular
        Game's mutable state, so it lives here rather than on Game.
        """
        # Unpack (width, depth, height)
        shape = self.board.cells.shape
        width, depth, height = (shape[0], shape[1], 1) if len(shape) == 2 else shape
        reshaped_grid = grid.reshape(width, depth, height)

        layer_strings = []

        # Iterate over height (3rd axis / index 2)
        for h in range(height):
            layer = reshaped_grid[:, :, h]  # (width, depth) horizontal slice
            padded = np.pad(layer, pad_width=2, constant_values=OUTSIDE_BOARD)
            lines = []

            if height > 1:
                visual_width = (width + 2) * 2
                lines.append(f"layer {h}".ljust(visual_width))

            # Iterate over depth rows (r) and width cols (c)
            for r in range(1, depth + 3):
                row_chars = []
                for c in range(1, width + 3):
                    cell = padded[c, r]

                    if cell == EMPTY:
                        row_chars.append('  ')
                    elif cell >= 0:
                        txt = f"{self.blocks[cell].letter} " if print_letters else "  "
                        row_chars.append(f"{self.blocks[cell].terminal_color}{txt}{Style.RESET_ALL}")
                    else:  # cell == OUTSIDE_BOARD
                        n, s = padded[c, r-1] != OUTSIDE_BOARD, padded[c, r+1] != OUTSIDE_BOARD
                        w, e = padded[c-1, r] != OUTSIDE_BOARD, padded[c+1, r] != OUTSIDE_BOARD
                        nw, ne = padded[c-1, r-1] != OUTSIDE_BOARD, padded[c+1, r-1] != OUTSIDE_BOARD
                        sw, se = padded[c-1, r+1] != OUTSIDE_BOARD, padded[c+1, r+1] != OUTSIDE_BOARD

                        if n and w:   char = '┏━'
                        elif n and e: char = '━┓'
                        elif s and w: char = '┗━'
                        elif s and e: char = '━┛'
                        elif n or s:  char = '━━'
                        elif w:       char = '┃ '
                        elif e:       char = ' ┃'
                        elif nw:      char = '┛ '
                        elif ne:      char = ' ┗'
                        elif sw:      char = '┓ '
                        elif se:      char = ' ┏'
                        else:         char = '  '

                        row_chars.append(char)

                lines.append("".join(row_chars))
            layer_strings.append(lines)

        for row_tuple in zip(*layer_strings):
            print('  '.join(row_tuple))


class Game:
    def __init__(self, setup: PuzzleSetup):
        self.setup = setup

        self.grid = self._fresh_grid()
        self.chosen_placement_idx = {idx: UNPLACED for idx in self.blocks.keys()}

    # board/blocks/placements are read through the shared setup rather than
    # duplicated as separate attributes, so there is exactly one place
    # that owns them.
    @property
    def board(self) -> Board:
        return self.setup.board

    @property
    def blocks(self) -> BlockCollection:
        return self.setup.blocks

    @property
    def placements(self) -> dict:
        return self.setup.placements

    def _fresh_grid(self) -> np.ndarray:
        """An empty grid: EMPTY on cells that are part of the board,
        OUTSIDE_BOARD everywhere else."""
        return np.where(self.board.cells, EMPTY, OUTSIDE_BOARD)

    def place(self, block_idx: int, placement_idx: int):
        """Place a block on the grid at the specified placement index."""
        placement_idcs = self.placements[block_idx][placement_idx]
        if not np.all(self.grid.flat[placement_idcs] == EMPTY):
            raise ValueError("Placement overlaps existing blocks.")
        if not self.chosen_placement_idx[block_idx] == UNPLACED:
            raise ValueError("Block is already placed.")
        self.place_unchecked(block_idx, placement_idx)

    def place_unchecked(self, block_idx: int, placement_idx: int):
        """Place a block without validating it first. Only safe when the
        caller has already independently guaranteed the placement doesn't
        overlap anything and the block isn't already placed -- e.g.
        Solver, whose own pruning makes place()'s checks redundant on
        every call in its search loop. Everyone else should use place()."""
        placement_idcs = self.placements[block_idx][placement_idx]
        self.grid.flat[placement_idcs] = block_idx
        self.chosen_placement_idx[block_idx] = placement_idx

    def remove(self, block_idx):
        """Remove a block from the grid, if present."""
        if self.chosen_placement_idx[block_idx] == UNPLACED:
            raise ValueError("Block is not currently placed.")
        self.remove_unchecked(block_idx)

    def remove_unchecked(self, block_idx):
        """Remove a block without checking it's actually placed first.
        Same caveat as place_unchecked: only safe when the caller already
        knows the block is placed."""
        placement_idx = self.chosen_placement_idx[block_idx]
        placement_idcs = self.placements[block_idx][placement_idx]
        self.grid.flat[placement_idcs] = EMPTY
        self.chosen_placement_idx[block_idx] = UNPLACED

    def clear(self):
        self.grid = self._fresh_grid()
        self.chosen_placement_idx = {idx: UNPLACED for idx in self.blocks}

    def copy(self, rename=None):
        # setup (board, blocks, placements) is immutable and shared by every
        # puzzle in the book, so it's safe - and much cheaper - to share it
        # by reference rather than deep-copying it. Only the actual
        # per-instance mutable state (grid, chosen placements) is copied.
        clone = copy.copy(self)
        clone.grid = self.grid.copy()
        clone.chosen_placement_idx = dict(self.chosen_placement_idx)
        if rename is not None:
            clone.name = rename
        return clone

    def solve(self, mode: int = 2, seed: int = None, disp: bool = False):
        """Solve this game/puzzle, yielding each solution as its own Game
        (or Puzzle) copy, fully placed -- grid and chosen_placement_idx
        both correct and consistent, same as any other Game."""
        solver = Solver(self)
        for sol_idx, sol in enumerate(solver.solve(mode=mode, seed=seed)):
            if hasattr(sol, "name"):
                sol.name = f"{self.name} (solution {sol_idx + 1})"
            if disp:
                print(f"Solution {sol_idx + 1}:")
                sol.print_to_terminal()
            yield sol

    def print_to_terminal(self, print_letters: bool = True):
        self.setup.print_to_terminal(self.grid, print_letters)


class Puzzle(Game):
    def __init__(self, setup: PuzzleSetup, letter_grid: np.ndarray, name: str = "", difficulty: str = "starter", empty: str = ' '):
        super().__init__(setup)
        self.name = name
        self.difficulty = difficulty
        self._initialize_grid(letter_grid, empty)

    def _initialize_grid(self, letter_grid: np.ndarray, empty: str) -> None:
        arr = np.asarray(letter_grid).T
        if arr.shape != self.grid.shape:
            raise ValueError(f"letter_grid is not the same shape as the board in puzzle '{self.name}'.")

        # Map letters to indices
        letter_to_idx = {b.letter: idx for idx, b in self.blocks.items()}

        # Check for unknown letters
        placed_mask = (arr != empty)
        unknown = set(arr[placed_mask]) - set(letter_to_idx)
        if unknown:
            raise ValueError(f"unknown letter(s) in grid: {unknown}")

        # Vectorized boundary check for all letters at once (no np.isin needed)
        if not np.all(self.board.cells[placed_mask]):
            raise ValueError(f"letter placed outside valid cell on the board in puzzle '{self.name}'")

        # Fill grid and validate placements
        for letter, idx in letter_to_idx.items():
            mask = (arr == letter)
            if not mask.any():
                continue

            self.grid[mask] = idx

            # Pure NumPy placement check: match 1D indices across rows
            placed_indices = np.flatnonzero(mask)
            valid_placements = self.placements[idx]  # Shape: (N, block_size)

            # Ensure cell count matches, then find which placement row (if any) matches
            matching_rows = (
                np.flatnonzero((valid_placements == placed_indices).all(axis=1))
                if placed_indices.shape[0] == valid_placements.shape[1]
                else np.empty(0, dtype=int)
            )

            if matching_rows.size == 0:
                raise ValueError(f"Block {letter} placement does not match any valid shape configuration in puzzle '{self.name}'.")

            # Record this block as placed so downstream consumers (e.g. Solver)
            # know it's fixed rather than free to place.
            self.chosen_placement_idx[idx] = int(matching_rows[0])


class PuzzleBook(UserDict):
    def __init__(self, *puzzles):
        _assert_unique(puzzles, lambda p: p.name, "puzzle name")
        # Every puzzle in a book is built from one shared PuzzleSetup (that's
        # the whole point -- see PuzzleSetup's docstring), so the book can
        # expose it directly instead of making callers reach into an
        # arbitrary puzzle's .setup themselves.
        self.setup: PuzzleSetup = _shared(puzzles, lambda p: p.setup, "PuzzleSetup")
        super().__init__({p.name: p for p in puzzles})

    @property
    def board(self) -> Board:
        return self.setup.board

    @property
    def blocks(self) -> BlockCollection:
        return self.setup.blocks