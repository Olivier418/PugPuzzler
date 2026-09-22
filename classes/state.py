import copy
import math

import numpy as np

from constants import EMPTY, UNPLACED
from .blocks import BlockCollection
from .boards import Board
from .setup import Setup
from .solver import Solver


class State:
    def __init__(self, setup: Setup):
        self.setup = setup

        # Where this State's originating Puzzle/PuzzleBook was loaded from
        # (a classes.source.Source), if it was loaded from disk at all.
        # Populated by serialization.loading.load_game; stays None for
        # anything constructed by hand. solving.solve_puzzle/solve_puzzlebook
        # read this to auto-save results to a mirrored solutions/ path.
        self.source = None

        # Overridden by Puzzle; a bare State is anonymous.
        self.name: str | None = None

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
    def placement_cells(self) -> dict:
        return self.setup.placement_cells

    def _fresh_grid(self) -> np.ndarray:
        """An empty grid: EMPTY on every one of the board's real cells.
        Compact -- shape (n_cells,) -- no OUTSIDE_BOARD cells are stored
        internally at all; those only reappear when expanding back to the
        full board shape (rendering, disk I/O)."""
        return np.full(self.setup.n_cells, EMPTY)

    def place(self, block_idx: int, placement_idx: int):
        """Place a block on the grid at the specified placement index."""
        placement_idcs = self.placement_cells[block_idx][placement_idx]
        if not np.all(self.grid[placement_idcs] == EMPTY):
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
        placement_idcs = self.placement_cells[block_idx][placement_idx]
        self.grid[placement_idcs] = block_idx
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
        placement_idcs = self.placement_cells[block_idx][placement_idx]
        self.grid[placement_idcs] = EMPTY
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

    def solve(
        self,
        seed: int = None,
        time_limit: float = math.inf,
        max_solutions: float = math.inf,
        up_to_symmetry: bool = False,
        **options,
    ):
        """Solve this state, yielding each solution as its own copy of it
        (a State, or a Puzzle), fully placed -- grid and
        chosen_placement_idx both correct and consistent. A named solution
        is renamed "<name> (solution <i>)". Printing progress is the
        caller's job (see solving.solve_puzzle's `verbose`).

        `seed` randomises the order solutions are found in (never which
        ones). Stops early once `time_limit` seconds have passed or
        `max_solutions` have been found, whichever comes first (both
        default to infinity).

        `up_to_symmetry` yields one solution per symmetry class instead of
        every solution -- so it changes the result rather than the route
        to it, which is why it is named here alongside `max_solutions`
        rather than left in `options` for benchmark.py to treat as a
        solver config.

        `options` (`symmetry`, `order`) are forwarded to Solver.solve
        as-is; they never change the solution set, only how fast and in
        what order it arrives."""
        solver = Solver(self)
        for sol_idx, sol in enumerate(
            solver.solve(
                seed=seed,
                time_limit=time_limit,
                max_solutions=max_solutions,
                up_to_symmetry=up_to_symmetry,
                **options,
            )
        ):
            if self.name:
                sol.name = f"{self.name} (solution {sol_idx + 1})"
            yield sol

    def _print_header(self) -> str:
        return f"State {self.name}" if self.name else "State"

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
        if arr.shape != self.board.cells.shape:
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

            # mask is full board-shaped (arr's shape); grid is compact, so
            # its cell indices are re-based through the same
            # flat_to_compact mapping placements were computed with.
            placed_indices = self.setup.flat_to_compact[np.flatnonzero(mask)]
            self.grid[placed_indices] = idx

            # Pure NumPy placement check: match 1D indices across rows
            valid_placements = self.placement_cells[idx]  # Shape: (N, block_size)

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
