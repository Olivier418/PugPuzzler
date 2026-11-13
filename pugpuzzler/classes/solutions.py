from typing import NamedTuple

import numpy as np

from .puzzle import Puzzle
from .setup import Setup


class SolveStats(NamedTuple):
    """How one solver run went -- a property of the run, not of the
    solutions, so kept apart from Solution. `options` = the solver keyword
    arguments besides the seed (`branch`, `order`), so benchmark.py can
    tell variants apart. `complete` = the search ran to the end, rather
    than being cut short by `time_limit`/`max_solutions`."""
    puzzle_name: str
    options: dict
    seed: int | None
    duration: float
    complete: bool
    elapsed: list[float]  # per-solution elapsed time, same order/index as the matching Solution.rows (if kept)


class Solution:
    """One Puzzle's solutions: the Puzzle plus every solution as a row (each
    block's placement index, in block order; see solver.solve_rows) -- the
    solver's own form, ~20x smaller than a grid. Grids exist only at the
    edges: `grids`, solutions.json and printing.

    The Puzzle is the live one (solved, or found in games/ by
    load_run_books), so name, difficulty, filled cells and Setup are read
    off it rather than copied here."""

    def __init__(self, puzzle: Puzzle, rows: np.ndarray):
        self.puzzle = puzzle
        self.rows = np.asarray(rows).reshape(-1, len(puzzle.blocks))

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def grids(self) -> np.ndarray:
        """Every solution as a full board-shaped int64 grid, (k, *board
        shape) -- built on each access (440 bytes a grid on IQpuzzler main)."""
        return self.setup.rows_to_grids(self.rows)

    @property
    def puzzle_name(self) -> str:
        return self.puzzle.name

    @property
    def setup(self) -> Setup:
        return self.puzzle.setup

    def __repr__(self) -> str:
        if not len(self):
            return f"Solution to puzzle {self.puzzle_name} (no results)"
        return "\n\n".join(
            self.puzzle.solved_copy(row, nr).render() for nr, row in enumerate(self.rows, start=1)
        )
