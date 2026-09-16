import copy
from collections import UserDict
import itertools

import numpy as np

from constants import EMPTY, OUTSIDE_BOARD, UNPLACED
from Solver import Solver

from ._utils import _assert_unique, _shared
from .blocks import Block, BlockCollection
from .boards import Board
from .rendering import render as _render


class Setup:
    """The fixed setup shared by every game: a board, a block
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
        board_cells = np.sum(self.board.cells)
        if total_cells != board_cells:
            raise ValueError(
                f"Total block cells ({total_cells}) do not match board cells ({board_cells})."
            )

        for idx, placements in self.placements.items():
            if len(placements) == 0:
                raise ValueError(f"Block {idx} has no valid placements on the board.")

    def render(self, grid: np.ndarray, header: str = None, leftover_idcs=None, print_letters: bool = True) -> str:
        """Build the text representation used by every __repr__ in this
        module: an optional header, then the board, and -- when
        `leftover_idcs` is given -- shape diagrams of those blocks laid
        out underneath the board. The actual rendering lives in
        `classes.rendering` (a display concern, not part of this class's
        board/blocks/placements model); this is a thin facade so callers
        can keep saying `setup.render(...)`.
        """
        return _render(self.board, self.blocks, grid, header=header, leftover_idcs=leftover_idcs, print_letters=print_letters)


class State:
    def __init__(self, setup: Setup):
        self.setup = setup

        # Where this State's originating Puzzle/PuzzleBook was loaded from
        # (a classes.source.Source), if it was loaded from disk at all.
        # Populated by serialization.loading.load_game; stays None for
        # anything constructed by hand. Solution.from_puzzle/from_puzzlebook
        # read this to auto-save results to a mirrored solutions/ path.
        self.source = None

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
                print(sol)
            yield sol

    def _print_header(self) -> str:
        name = getattr(self, "name", None)
        return f"State {name}" if name else "State"

    def render(self, show_leftover: bool = True) -> str:
        """Text representation used by __repr__: header + grid, plus
        shape diagrams of any still-unplaced blocks underneath it.
        `show_leftover=False` is used by PuzzleBook/Game when printing
        every puzzle in a book, where the per-puzzle legend is just
        noise."""
        leftover = None
        if show_leftover:
            leftover = [idx for idx, p in self.chosen_placement_idx.items() if p == UNPLACED]
        return self.setup.render(self.grid, header=self._print_header(), leftover_idcs=leftover)

    def __repr__(self) -> str:
        return self.render()


class Puzzle(State):
    def __init__(self, setup: Setup, letter_grid: np.ndarray = None, name: str = "", difficulty: str = None, empty: str = ' '):
        super().__init__(setup)
        self.name = name
        self.difficulty = difficulty
        if letter_grid is not None:
            self._initialize_grid(letter_grid, empty)

    def _initialize_grid(self, letter_grid: np.ndarray, empty: str) -> None:
        # Internal convention (see classes.rendering.grid_lines): grids are
        # shaped (width, depth). letter_grid, like boards.json's "cells",
        # is authored the human-readable way -- one row per depth
        # position, `width` entries per row, i.e. (depth, width) as
        # written -- so it gets the same one-time transpose at this JSON
        # boundary. See serialization.loading._cells_from_json for the
        # board-cells counterpart of this exact rule.
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

    def _print_header(self) -> str:
        return f"Puzzle {self.name}"


class PuzzleBook(UserDict):
    def __init__(self, *puzzles, name: str = None):
        _assert_unique(puzzles, lambda p: p.name, "puzzle name")
        # Every puzzle in a book is built from one shared PuzzleSetup (that's
        # the whole point -- see PuzzleSetup's docstring), so the book can
        # expose it directly instead of making callers reach into an
        # arbitrary puzzle's .setup themselves.
        self.setup: Setup = _shared(puzzles, lambda p: p.setup, "PuzzleSetup")
        self.name = name

        # Same idea as State.source: where this book was loaded from, if
        # anywhere. Populated by serialization.loading.load_game.
        self.source = None

        super().__init__({p.name: p for p in puzzles})

    @property
    def board(self) -> Board:
        return self.setup.board

    @property
    def blocks(self) -> BlockCollection:
        return self.setup.blocks

    def __repr__(self) -> str:
        header = f"Puzzle Book {self.name}" if self.name else "Puzzle Book"
        body = "\n\n".join(p.render(show_leftover=False) for p in self.values())
        return f"{header}\n\n{body}" if body else header