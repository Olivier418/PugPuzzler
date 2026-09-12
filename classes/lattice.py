from functools import cached_property
import itertools
import numpy as np


class Lattice:
    """Describes the local geometry of a lattice of unit cells.

    The only information needed to fully pin down one of these
    rectangular/triangular-type lattices is, for every pair of axes,
    whether that pair is offset by 1/2 from one another (e.g. a
    triangular lattice) or not (i.e. orthogonal). This is captured as a
    symmetric, zero-diagonal boolean adjacency matrix over the axes.

    From that adjacency matrix alone, everything else the placement
    search needs can be derived:
      - gram2: the (2x the) Gram matrix of the lattice basis vectors,
        used to test distances/orthogonality in pure integer arithmetic.
      - unit_vectors: every integer combination of basis vectors that
        has unit length - i.e. every direction a block can step in.
      - neighbor_structure: the boolean stencil of which grid cells
        (offset by -1/0/1 along each axis) count as adjacent.

    Only lattices with at most 3 axes are supported. Beyond 3 axes, an
    arbitrary offset adjacency matrix is no longer guaranteed to
    correspond to a valid (positive-definite) lattice - some offset
    graphs would imply axes with zero or negative "length", which is
    physically meaningless - and validating that would require an
    explicit eigenvalue check. Every possible offset graph on <=3 axes
    happens to already be valid, so that check is intentionally omitted
    here rather than carried as dead code.
    """

    MAX_NDIM = 3

    def __init__(self, offset_adjacency: np.ndarray):
        offset_adjacency = np.asarray(offset_adjacency, dtype=bool)

        if offset_adjacency.ndim != 2 or offset_adjacency.shape[0] != offset_adjacency.shape[1]:
            raise ValueError("offset_adjacency must be a square matrix.")
        if not np.array_equal(offset_adjacency, offset_adjacency.T):
            raise ValueError("offset_adjacency must be symmetric.")
        if np.any(np.diag(offset_adjacency)):
            raise ValueError("offset_adjacency must have a zero diagonal: an axis cannot be offset from itself.")

        ndim = offset_adjacency.shape[0]
        if ndim > self.MAX_NDIM:
            raise ValueError(
                f"Lattice only supports up to {self.MAX_NDIM} axes, got {ndim}. "
                f"Beyond {self.MAX_NDIM} axes, an offset adjacency matrix is no "
                "longer guaranteed to correspond to a valid lattice."
            )

        self.ndim = ndim
        self.offset_adjacency = offset_adjacency

    @cached_property
    def gram2(self) -> np.ndarray:
        """Twice the Gram matrix of the lattice basis vectors, kept as
        pure integers. Diagonal entries are always 2 (unit-length basis
        vectors); off-diagonal entries are 2 * (1/2) = 1 where axes are
        offset, and 0 where they are orthogonal.
        """
        return 2 * np.eye(self.ndim, dtype=int) + self.offset_adjacency.astype(int)

    @cached_property
    def unit_vectors(self) -> list[np.ndarray]:
        """All integer vectors v in {-1,0,1}^ndim with v^T @ gram2 @ v == 2,
        i.e. every direction of unit length reachable along the lattice.
        """
        gram2 = self.gram2
        vectors = []
        for v_tuple in itertools.product([-1, 0, 1], repeat=self.ndim):
            if v_tuple == (0,) * self.ndim:
                continue
            v = np.array(v_tuple, dtype=int)
            if v @ gram2 @ v == 2:
                vectors.append(v)
        return vectors

    @cached_property
    def neighbor_structure(self) -> np.ndarray:
        """Boolean (3,)*ndim stencil marking which offsets (in -1/0/1 per
        axis, indexed 0/1/2) count as adjacent to the center cell.
        """
        structure = np.zeros((3,) * self.ndim, dtype=bool)
        structure[(1,) * self.ndim] = True
        for v in self.unit_vectors:
            structure[tuple(v + 1)] = True
        return structure