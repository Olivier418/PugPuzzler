import itertools
from functools import cached_property

import numpy as np

from constants import OUTSIDE_BOARD
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
        self.placement_cells = self._compute_placement_indices()
        self._validate()

        # Compact indexing: internally, grids only ever cover the board's
        # real cells (no OUTSIDE_BOARD dead weight -- can be ~half the
        # array on a PyramidBoard). compact_to_flat/flat_to_compact map
        # between that dense 0..n_cells-1 range and the full board-shaped
        # flat indexing placements were originally computed in.
        cells_flat = self.board.cells.ravel()
        self.compact_to_flat = np.flatnonzero(cells_flat)
        self.n_cells = self.compact_to_flat.size
        self.flat_to_compact = np.full(cells_flat.size, -1, dtype=np.int64)
        self.flat_to_compact[self.compact_to_flat] = np.arange(self.n_cells)

        # Placements are computed above as flat indices into the full
        # board shape; re-base them into compact space once here so every
        # other consumer (State.grid, Solver) only ever deals with the
        # dense range.
        self.placement_cells = {
            idx: self.flat_to_compact[flat]
            for idx, flat in self.placement_cells.items()
        }

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

        for idx, placements in self.placement_cells.items():
            if len(placements) == 0:
                raise ValueError(f"Block {idx} has no valid placements on the board.")

    # ---- symmetry -------------------------------------------------------
    # Pure geometry, all lazily built: a solve that doesn't ask for
    # symmetry (symmetry=False without up_to_symmetry) never pays for any
    # of it. Everything here is shared by reference across every Puzzle in
    # a book, like placements.

    @cached_property
    def cell_coords(self) -> np.ndarray:
        """(n_cells, ndim) coordinates of every compact cell, i.e. the
        compact index -> board position map that the symmetry tables are
        built from."""
        return np.stack(
            np.unravel_index(self.compact_to_flat, self.board.cells.shape), axis=1
        )

    @cached_property
    def _identity_perm(self) -> np.ndarray:
        """The do-nothing cell permutation, shared rather than rebuilt:
        region_symmetries returns it on the overwhelmingly common
        no-symmetry path, and nothing ever mutates a returned image."""
        return np.arange(self.n_cells, dtype=np.int64)

    @cached_property
    def _point_group_coords(self) -> np.ndarray:
        """(K, n_cells, ndim): every cell's coordinates under every matrix
        of the lattice's point group, precomputed so region_symmetries
        never has to do coordinate arithmetic at all -- it just gathers
        the rows it wants. Tiny (48 x 55 x 3 on the pyramid) and pure
        board geometry."""
        return np.einsum(
            "kij,nj->kni", self.board.lattice.point_group, self.cell_coords
        )

    @cached_property
    def _symmetry_tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Flattens the whole symmetry search into integer table lookups,
        so region_symmetries does no coordinate arithmetic at all. Returns
        `(transformed_flat, cell_flat, lookup)`.

        Everything is indexed into a *padded* grid big enough that any
        point-group image of any sub-region, translated anywhere it could
        plausibly land, still falls inside it. That matters because
        flattening is only linear -- `flat(a) - flat(b) == flat(a - b)`,
        and comparing flat values is comparing coordinates
        lexicographically -- while nothing wraps around an axis. The
        padding is derived from the actual extremes of
        _point_group_coords, so it is exactly as large as it needs to be.

        - `transformed_flat[k][i]`: cell i's coordinates under matrix k,
          flattened. Only ever used in differences, so its origin is
          arbitrary.
        - `cell_flat[i]`: cell i's own coordinates, flattened.
        - `lookup[f]`: the compact cell at padded flat index f, or
          `n_cells` -- one past the end -- for anything that isn't one of
          the board's cells, so an off-board hit indexes a sentinel slot
          instead of needing its own test.
        """
        transformed = self._point_group_coords
        low = transformed.min(axis=(0, 1))
        pad = transformed.max(axis=(0, 1)) - low

        padded_shape = np.asarray(self.board.cells.shape) + 2 * pad
        strides = np.ones(self.board.ndim, dtype=np.int64)
        for axis in range(self.board.ndim - 2, -1, -1):
            strides[axis] = strides[axis + 1] * padded_shape[axis + 1]

        transformed_flat = (transformed - low) @ strides
        cell_flat = (self.cell_coords + pad) @ strides

        lookup = np.full(int(np.prod(padded_shape)), self.n_cells, dtype=np.int64)
        lookup[cell_flat] = np.arange(self.n_cells)
        return transformed_flat, cell_flat, lookup

    def region_symmetries(self, region: np.ndarray) -> list[np.ndarray]:
        """Every symmetry of an arbitrary sub-region of the board's cells,
        as a permutation of compact cell indices: image[i] is the compact
        index cell i maps to. Each returned image is the **identity
        outside the region**, so applying one to a whole grid only ever
        rearranges the region itself. Always includes the identity.

        `region` is a compact (n_cells,) bool mask. Passing an all-True
        mask gives the whole board's symmetries (see board_symmetries);
        passing the still-open cells of a partly-filled puzzle gives that
        leftover shape's own symmetries, which are generally *not*
        restrictions of any whole-board symmetry -- a symmetric pocket
        left in an otherwise asymmetric board.

        Only the lattice's point group has to be searched, not point
        group x translations: a finite region admits no nontrivial
        translational self-symmetry (composing two candidate translations
        for the same matrix gives a pure translation stabilising a finite
        set, which must be zero), so for each matrix M there is at most
        one translation t with M(region) + t == region. Lining up the two
        shapes' lexicographically smallest cells recovers it directly.

        Solver calls this exactly once per solve, on the root's open
        cells; every group it uses deeper is a subgroup of that one,
        obtained by filtering rather than by searching again.
        """
        idx = np.flatnonzero(region)
        if idx.size <= 1:
            return [self._identity_perm]

        transformed_flat, cell_flat, lookup = self._symmetry_tables

        # (K, m) flat positions of the region's cells under every matrix,
        # then shifted so each image's lexicographically smallest cell
        # sits on the region's own. That shift is the only translation
        # that can possibly work -- see the note on uniqueness above --
        # and idx is ascending, so idx[0] is that smallest cell.
        image = transformed_flat[:, idx].copy()
        image -= image.min(axis=1, keepdims=True)
        image += cell_flat[idx[0]]

        # A matrix is a symmetry iff every one of its images lands inside
        # the region -- no need to compare cell sets. The map is injective
        # (the matrix is invertible, the translation and the lookup are
        # one-to-one), so m distinct cells inside a region of m cells can
        # only be all of it. `inside` carries one extra False so that the
        # off-board sentinel above falls through it.
        inside = np.append(region, False)
        compact = lookup[image]
        rows = compact[inside[compact].all(axis=1)]

        # Distinct matrices can act identically on a small region (every
        # symmetry fixes a single cell, say), and the solver pays per
        # image, so collapse those here -- comparing only the region's own
        # cells, since the rest is the identity either way.
        seen = set()
        distinct = []
        for row in rows:
            key = row.tobytes()
            if key not in seen:
                seen.add(key)
                distinct.append(row)

        # A lone survivor can only be the identity (the images form a
        # group), which is the overwhelmingly common case -- hand back the
        # shared one rather than building a copy per call.
        if len(distinct) == 1:
            return [self._identity_perm]

        images = []
        for row in distinct:
            image = self._identity_perm.copy()
            image[idx] = row
            images.append(image)
        return images

    @cached_property
    def board_symmetries(self) -> list[np.ndarray]:
        """The whole board's own symmetries -- region_symmetries applied
        to every cell. Geometry-only (independent of blocks or
        placements). Solver works from the open region instead, so this
        is for callers that want the board's group on its own."""
        return self.region_symmetries(np.ones(self.n_cells, dtype=bool))

    @cached_property
    def placement_lookup(self) -> dict:
        """{block_idx: {sorted cell indices as bytes -> placement_idx}},
        i.e. "which placement of this block covers exactly these cells".

        Lets a symmetry be applied to a placement: transform its cells,
        look the result back up. Guaranteed to hit for a symmetry of a
        region the placement lies inside -- a lattice isometry carries a
        valid placement to an isometric copy of the same block, still on
        the board, and _valid_orientations / _compute_placement_indices
        enumerate every on-board isometric copy of every block."""
        return {
            idx: {np.sort(cells).tobytes(): i for i, cells in enumerate(arr)}
            for idx, arr in self.placement_cells.items()
        }

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
