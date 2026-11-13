import itertools
from functools import cached_property

import numpy as np

from pugpuzzler.constants import EMPTY, OUTSIDE_BOARD
from . import kernel
from .blocks import Block, BlockCollection
from .boards import Board
from .rendering import render as _render


class Setup:
    """A board, a block collection, and every valid placement of each block
    on that board (expensive to compute). None of it depends on a puzzle's
    pre-filled cells, so it is computed once and shared by reference by
    every Puzzle on that board."""

    def __init__(self, blocks: BlockCollection, board: Board):
        self.board = board
        self.blocks = blocks
        self._validate_area()

        # Compact indexing: grids only cover the board's real cells (no
        # OUTSIDE_BOARD cells -- over half the array on a PyramidBoard).
        # compact_to_flat/flat_to_compact map between 0..n_cells-1 and flat
        # indices into the full board shape.
        cells_flat = self.board.cells.ravel()
        self.compact_to_flat = np.flatnonzero(cells_flat)
        self.n_cells = self.compact_to_flat.size
        self.flat_to_compact = np.full(cells_flat.size, -1, dtype=np.int64)
        self.flat_to_compact[self.compact_to_flat] = np.arange(self.n_cells)

        # One (N, block size) array of compact cell indices per block, in
        # block order (computed as flat indices, re-based once here).
        self.placement_cells = [self.flat_to_compact[flat] for flat in self._compute_placement_indices()]

    def _valid_orientations(self, block: Block) -> list[np.ndarray]:
        """Every distinct orientation of `block` on this board, as board
        coordinates shifted to start at 0 and sorted."""
        k = block.ndim

        G_board2 = self.board.gram2
        # A block's own coordinates are always on an orthogonal grid, whose
        # doubled Gram matrix is 2I (see Board.gram2).
        G_piece2 = 2 * np.eye(k, dtype=int)

        candidate_vecs = self.board.unit_vectors

        # Every map M from the block's axes onto board unit vectors that
        # preserves distances is an orientation.
        unique_shapes = []
        seen = set()

        for cols in itertools.product(candidate_vecs, repeat=k):
            M = np.column_stack(cols)  # Shape (d, k)

            # distance-preserving, in integers: M^T @ (2G_board) @ M == 2G_piece
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

    # ---- derived tables (lazy, shared like placement_cells) -------------

    @cached_property
    def kernel_tables(self) -> kernel.Tables:
        """The search's static tables (see kernel.Tables): ~70 KB, ~10 ms
        to build. A book is up to 72 puzzles on one Setup, so caching them
        here rather than per puzzle is worth ~2x on a whole-book solve."""
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
        """Pay a first solve's one-off costs up front (build kernel_tables,
        load the compiled kernel; see kernel.warmup): call before starting a
        timer. The JIT part is per process, so repeat calls are free."""
        kernel.warmup(self.kernel_tables)

    def render(
        self,
        grid: np.ndarray,
        header: str = None,
        leftover_idcs=None,
        labels=None,
        cell_width: int = 2,
        letters: bool = True,
        mark: int = None,
    ) -> str:
        """rendering.render on compact arguments: `grid` and `labels` are
        compact and expanded to the full board shape here, `mark` is a
        compact cell index; the rest pass through (see rendering.grid_lines)."""
        full_labels = None
        if labels is not None:
            full_labels = np.full(self.board.cells.size, '', dtype=object)
            full_labels[self.compact_to_flat] = labels
            full_labels = full_labels.reshape(self.board.cells.shape)
        full_mark = None
        if mark is not None:
            full_mark = np.unravel_index(self.compact_to_flat[mark], self.board.cells.shape)
        return _render(self.board, self.blocks, self.to_full_grid(grid), header=header,
                       leftover_idcs=leftover_idcs, labels=full_labels, cell_width=cell_width,
                       letters=letters, mark=full_mark)

    def to_full_grid(self, grid: np.ndarray) -> np.ndarray:
        """A compact (n_cells,) grid as a full board-shaped array,
        OUTSIDE_BOARD everywhere else (for rendering and writing to disk)."""
        full = np.full(self.board.cells.size, OUTSIDE_BOARD, dtype=grid.dtype)
        full[self.compact_to_flat] = grid
        return full.reshape(self.board.cells.shape)
