import itertools
from functools import cached_property

import numpy as np

from constants import EMPTY, OUTSIDE_BOARD
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
        # consumer (Puzzle.grid, the solver) only ever deals with the dense
        # range. One (N, block size) array per block, in block order.
        self.placement_cells = [self.flat_to_compact[flat] for flat in self._compute_placement_indices()]

    def _valid_orientations(self, block: Block) -> list[np.ndarray]:
        k = block.ndim

        G_board2 = self.board.gram2
        # A block's own coordinates are always on an orthogonal grid, whose
        # doubled Gram matrix is 2I (see Board.gram2).
        G_piece2 = 2 * np.eye(k, dtype=int)

        # 1. All unit vectors in board space, i.e. v^T @ (2G) @ v == 2.
        # Cached on the board, so this is computed once per board and
        # reused across every block, rather than recomputed each time.
        candidate_vecs = self.board.unit_vectors

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

    def _compute_placement_indices(self) -> list[np.ndarray]:
        result = []
        cells = self.board.cells
        board_shape = cells.shape
        ndim = cells.ndim

        for block in self.blocks.values():
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

            if not result_chunks:
                raise ValueError(f"Block {block.letter} has no valid placements on the board.")
            result.append(np.concatenate(result_chunks, axis=0))
        return result

    def _validate_area(self):
        """Cheap check, run before the expensive placement enumeration."""
        total_cells = sum(b.count for b in self.blocks.values())
        board_cells = np.sum(self.board.cells)
        if total_cells != board_cells:
            raise ValueError(
                f"Total block cells ({total_cells}) do not match board cells ({board_cells})."
            )

    # ---- derived tables -------------------------------------------------
    # Lazily built and then shared by reference across every Puzzle in a
    # book, exactly like placement_cells above. A Setup that is only ever
    # printed pays for none of it.

    @cached_property
    def kernel_tables(self) -> kernel.Tables:
        """Everything the search needs that depends on the board and blocks
        alone, and nothing that depends on which cells a particular puzzle
        starts with. ~70 KB, ~7 ms to build -- and a book is 72 puzzles on
        one Setup, so caching it here rather than per puzzle is worth ~2x on
        a whole-book solve."""
        return kernel.build_tables(self.placement_cells, self.n_cells)

    @cached_property
    def row_dtype(self) -> type:
        """The smallest dtype that holds every placement index: the dtype
        solution rows (see solver.solve_rows) are stored in."""
        most = max(len(p) for p in self.placement_cells)
        return np.uint16 if most <= np.iinfo(np.uint16).max else np.int32

    def rows_to_grids(self, rows: np.ndarray) -> np.ndarray:
        """Solution rows (see solver.solve_rows: one row per solution, every
        block's placement index in block order) as full board-shaped grids,
        shape (k, *board shape): exactly what to_full_grid gives for the
        same solutions' Puzzle.grid, dtype included."""
        rows = np.asarray(rows).reshape(-1, len(self.blocks))
        k = len(rows)
        compact = np.full((k, self.n_cells), EMPTY, dtype=np.int64)
        which = np.arange(k)[:, None]
        for idx in self.blocks:
            compact[which, self.placement_cells[idx][rows[:, idx]]] = idx
        full = np.full((k, self.board.cells.size), OUTSIDE_BOARD, dtype=np.int64)
        full[:, self.compact_to_flat] = compact
        return full.reshape(k, *self.board.cells.shape)

    def grids_to_rows(self, grids: np.ndarray) -> np.ndarray:
        """Inverse of rows_to_grids: full board-shaped grids of block
        indices, (k, *board shape), as solution rows (k, n_blocks) in
        `row_dtype`. Raises ValueError unless every grid is a solution:
        each block covering exactly the cells of one of its placements."""
        grids = np.asarray(grids).reshape(-1, *self.board.cells.shape)
        k = len(grids)
        compact = grids.reshape(k, -1)[:, self.compact_to_flat]
        rows = np.empty((k, len(self.blocks)), dtype=self.row_dtype)
        for idx, block in self.blocks.items():
            # np.nonzero walks row-major, so each solution's cells come out
            # together and ascending: one (k, size) array
            which, cells = np.nonzero(compact == idx)
            if not np.array_equal(np.bincount(which, minlength=k), np.full(k, block.count)):
                raise ValueError(f"Block {block.letter} doesn't cover exactly {block.count} cells in every grid.")
            # a placement is identified by its sorted cells, as one integer
            dims = (self.n_cells,) * block.count
            keys = np.ravel_multi_index(cells.reshape(k, block.count).T, dims)
            placement_keys = np.ravel_multi_index(np.sort(self.placement_cells[idx], axis=1).T, dims)
            order = np.argsort(placement_keys)
            pos = np.searchsorted(placement_keys, keys, sorter=order).clip(max=len(order) - 1)
            if not np.array_equal(placement_keys[order[pos]], keys):
                raise ValueError(f"Block {block.letter} isn't on one of its placements in every grid.")
            rows[:, idx] = order[pos]
        return rows

    def warmup(self):
        """Get everything a first solve would pay for up front: build
        kernel_tables and load the compiled kernel (see kernel.warmup).
        Call before starting a timer. The JIT part is per process, not
        per Setup, so repeat calls on any Setup are free."""
        kernel.warmup(self.kernel_tables)

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
