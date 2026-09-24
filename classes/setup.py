import itertools
from functools import cached_property

import numpy as np

from constants import OUTSIDE_BOARD
from . import kernel
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
        self._validate_area()

        # Compact indexing: internally, grids only ever cover the board's
        # real cells (no OUTSIDE_BOARD dead weight -- can be ~half the
        # array on a PyramidBoard). compact_to_flat/flat_to_compact map
        # between that dense 0..n_cells-1 range and the full board-shaped
        # flat indexing placements are computed in.
        cells_flat = self.board.cells.ravel()
        self.compact_to_flat = np.flatnonzero(cells_flat)
        self.n_cells = self.compact_to_flat.size
        self.flat_to_compact = np.full(cells_flat.size, -1, dtype=np.int64)
        self.flat_to_compact[self.compact_to_flat] = np.arange(self.n_cells)

        # Placements are computed as flat indices into the full board
        # shape and re-based into compact space once here, so every other
        # consumer (State.grid, Solver) only ever deals with the dense
        # range.
        self.placement_cells = {
            idx: self.flat_to_compact[flat]
            for idx, flat in self._compute_placement_indices().items()
        }
        self._validate_placements()

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

    def _validate_area(self):
        """Cheap check, run before the expensive placement enumeration."""
        total_cells = sum(b.count for b in self.blocks.values())
        board_cells = np.sum(self.board.cells)
        if total_cells != board_cells:
            raise ValueError(
                f"Total block cells ({total_cells}) do not match board cells ({board_cells})."
            )

    def _validate_placements(self):
        for idx, placements in self.placement_cells.items():
            if len(placements) == 0:
                raise ValueError(f"Block {idx} has no valid placements on the board.")

    # ---- derived tables -------------------------------------------------
    # Lazily built and then shared by reference across every Puzzle in a
    # book, exactly like placement_cells above. A Setup that is only ever
    # printed pays for none of it.

    @cached_property
    def cell_coords(self) -> np.ndarray:
        """(n_cells, ndim) coordinates of every compact cell, i.e. the
        compact index -> board position map. Its one consumer is
        kernel_tables below, which derives cell adjacency from it for the
        solver's "pockets" ranking rule."""
        return np.stack(
            np.unravel_index(self.compact_to_flat, self.board.cells.shape), axis=1
        )

    @cached_property
    def kernel_tables(self) -> tuple:
        """`(kernel.Tables, block_ids)`: everything the search needs that
        depends on the board and blocks alone, and nothing that depends on
        which cells a particular puzzle starts with. ~70 KB, ~7 ms to
        build -- and a book is 72 puzzles on one Setup, so caching it here
        rather than per puzzle is worth ~2x on a whole-book solve.

        Lives on the Setup for the same reason placement_cells does: it is
        derived, immutable data *about this setup*. The search's mutable
        state (kernel.Workspace) deliberately does not -- that is
        allocated per kernel entry, which is what makes a reused Solver
        reproducible."""
        return kernel.build_tables(
            self.placement_cells, self.n_cells,
            cell_coords=self.cell_coords,
            unit_vectors=self.board.lattice.unit_vectors,
        )

    def warmup(self):
        """Get everything a first solve would pay for up front: build
        kernel_tables and load the compiled kernel (see kernel.warmup).
        Call before starting a timer. The JIT part is per process, not
        per Setup, so repeat calls on any Setup are free."""
        kernel.warmup(self.kernel_tables[0])

    def render(self, grid: np.ndarray, header: str = None, leftover_idcs=None) -> str:
        """Build the text representation used by every __repr__ in this
        module: an optional header, then the board, and -- when
        `leftover_idcs` is given -- shape diagrams of those blocks laid
        out underneath the board. The actual rendering lives in
        `classes.rendering` (a display concern, not part of this class's
        board/blocks/placements model); this is a thin facade so callers
        can keep saying `setup.render(...)`.

        `grid` is compact (see __init__); rendering needs the real
        board shape (with OUTSIDE_BOARD filled back in) to draw the
        board's silhouette, so it's expanded here at this one boundary.
        """
        return _render(self.board, self.blocks, self.to_full_grid(grid), header=header, leftover_idcs=leftover_idcs)

    def to_full_grid(self, grid: np.ndarray) -> np.ndarray:
        """Scatter a compact (n_cells,) grid back into a full board-shaped
        array, OUTSIDE_BOARD everywhere else. Used at the rendering and
        disk-serialization boundaries -- the only places that need the
        real board shape back."""
        full = np.full(self.board.cells.size, OUTSIDE_BOARD, dtype=grid.dtype)
        full[self.compact_to_flat] = grid
        return full.reshape(self.board.cells.shape)

    def to_compact_grid(self, full_grid: np.ndarray) -> np.ndarray:
        """Inverse of to_full_grid(): pull a full board-shaped grid (e.g. read
        from disk, or a JSON letter_grid) down to compact space."""
        return full_grid.ravel()[self.compact_to_flat]
