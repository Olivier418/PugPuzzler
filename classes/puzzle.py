import copy

import numpy as np

from constants import EMPTY, UNPLACED
from .blocks import BlockCollection
from .boards import Board
from .setup import Setup
from .solver import solve_rows


class Puzzle:
    """A partly (or fully) placed board over a shared Setup: which block
    covers each cell (`grid`) and which placement each block is on
    (`chosen_placement_idx`, UNPLACED for a free block). Optionally built
    from a letter grid, as a book's puzzles are."""

    def __init__(self, setup: Setup, letter_grid: np.ndarray = None, name: str = "", difficulty: str = None):
        self.setup = setup
        self.name = name
        self.difficulty = difficulty

        # Where this Puzzle was loaded from (a classes.source.Source), set by
        # the serialization.games_io loaders; None for one built by hand.
        # solving.solve_puzzle/solve_puzzlebook save results under it.
        self.source = None

        # Compact -- shape (n_cells,), EMPTY on every one of the board's
        # real cells: no OUTSIDE_BOARD cells are stored internally at all;
        # those only reappear when expanding back to the full board shape
        # (rendering, disk I/O).
        self.grid = np.full(self.setup.n_cells, EMPTY)
        self.chosen_placement_idx = {idx: UNPLACED for idx in self.blocks}
        if letter_grid is not None:
            self._initialize_grid(letter_grid)

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
    def placement_cells(self) -> list:
        return self.setup.placement_cells

    @property
    def nr_filled_cells(self) -> int:
        return int((self.grid != EMPTY).sum())

    def _initialize_grid(self, letter_grid: np.ndarray) -> None:
        # Internal convention (see classes.rendering.grid_lines): grids are
        # shaped (width, depth). letter_grid, like boards.json's "cells",
        # is authored the human-readable way -- one row per depth
        # position, `width` entries per row, i.e. (depth, width) as
        # written -- so it gets the same one-time transpose at this JSON
        # boundary. See serialization.games_io._parse_board for the
        # board-cells counterpart of this exact rule.
        arr = np.asarray(letter_grid).T
        if arr.shape != self.board.cells.shape:
            raise ValueError(f"letter_grid is not the same shape as the board in puzzle '{self.name}'.")

        letter_to_idx = {b.letter: idx for idx, b in self.blocks.items()}
        placed_mask = arr != " "
        unknown = set(arr[placed_mask]) - set(letter_to_idx)
        if unknown:
            raise ValueError(f"unknown letter(s) in grid: {unknown}")
        if not np.all(self.board.cells[placed_mask]):
            raise ValueError(f"letter placed outside valid cell on the board in puzzle '{self.name}'")

        for letter, idx in letter_to_idx.items():
            mask = arr == letter
            if not mask.any():
                continue
            # mask is full board-shaped; grid is compact, so its cell indices
            # are re-based through the same flat_to_compact mapping placements
            # were computed with (both ascending, so rows compare directly).
            placed_indices = self.setup.flat_to_compact[np.flatnonzero(mask)]
            valid_placements = self.placement_cells[idx]  # (N, block size)
            matching_rows = (
                np.flatnonzero((valid_placements == placed_indices).all(axis=1))
                if placed_indices.shape[0] == valid_placements.shape[1]
                else np.empty(0, dtype=int)
            )
            if matching_rows.size == 0:
                raise ValueError(f"Block {letter} placement does not match any valid shape configuration in puzzle '{self.name}'.")
            self.place_unchecked(idx, int(matching_rows[0]))

    def place_unchecked(self, block_idx: int, placement_idx: int):
        """Place a block without validating it first: the caller guarantees
        the placement overlaps nothing and the block isn't placed yet (a
        solution row, or a bound's realization, always does). Writes the
        grid and chosen_placement_idx together, so the two can't disagree."""
        self.grid[self.placement_cells[block_idx][placement_idx]] = block_idx
        self.chosen_placement_idx[block_idx] = placement_idx

    def solved_copy(self, row, nr: int = None) -> "Puzzle":
        """A copy of this puzzle with every still-unplaced block placed as
        the solution row `row` says (every block's placement index in
        block order, as solve_rows yields them). Named "<name> (solution
        <nr>)" when this puzzle is named and `nr` is given."""
        solved = self.copy(rename=f"{self.name} (solution {nr})" if self.name and nr else None)
        for idx, placement_idx in zip(self.blocks, row):
            if solved.chosen_placement_idx[idx] == UNPLACED:
                solved.place_unchecked(idx, int(placement_idx))
        return solved

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

    def solve_rows(self, **options):
        """Every solution of this puzzle as a row, lazily; the generator
        returns whether the search ran to the end. See solver.solve_rows
        for the options (`seed`, `time_limit`, `max_solutions`, `order`,
        `branch`). Printing progress is the caller's job (see
        solving.solve_puzzle's `verbose`)."""
        return solve_rows(self, **options)

    def render(self, show_leftover: bool = True) -> str:
        """Text representation used by __repr__: header + grid, plus
        shape diagrams of any still-unplaced blocks underneath it.
        `show_leftover=False` is used by PuzzleBook when printing every
        puzzle in a book, where the per-puzzle legend is just noise."""
        leftover = None
        if show_leftover:
            leftover = [idx for idx, p in self.chosen_placement_idx.items() if p == UNPLACED]
        header = f"Puzzle {self.name}" if self.name else "Puzzle"
        return self.setup.render(self.grid, header=header, leftover_idcs=leftover)

    def __repr__(self) -> str:
        return self.render()
