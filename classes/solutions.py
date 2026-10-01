from typing import NamedTuple

import numpy as np

from .puzzle import Puzzle
from .setup import Setup


class SolveStats(NamedTuple):
    """Metadata about the solver run that produced a Solution -- kept
    separate from Solution itself since the solver is exhaustive: the
    solutions are deterministic, but how long it took / which solver
    options+seed were used is a property of the run, not of the
    solutions. `options` is whatever keyword arguments the solver was
    given besides the seed (`branch`, `order`; kept so runs of different
    solver variants can be told apart and compared by benchmark.py).

    `complete` says whether the run found every solution: True if the
    search ran to the end, False if `time_limit`/`max_solutions` cut it
    short (see solver.solve_rows' return value)."""
    puzzle_name: str
    options: dict
    seed: int | None
    duration: float
    complete: bool
    elapsed: list[float]  # per-solution elapsed time, same order/index as the matching Solution.rows (if kept)


class Solution:
    """One Puzzle's solutions: the Puzzle itself plus every solution as a
    row -- the placement index of every block, in the Setup's block order
    (see solver.solve_rows), the solver's own form and ~20x smaller than a
    grid. Grids exist only at the edges: `grids` builds them on demand
    (full board-shaped, see Setup.rows_to_grids), solutions.json stores
    them as letters, and printing renders them.

    The Puzzle is always live -- the one solving.py solved, or the one
    serialization.load_run_books found in games/ -- so everything
    about it (name, difficulty, filled cells, Setup) is read off it rather
    than copied here, where it could drift.

    A book's solutions (or stats) are just a dict keyed by puzzle name.
    Producing one (solving + saving) lives in `solving.py`."""

    def __init__(self, puzzle: Puzzle, rows: np.ndarray):
        self.puzzle = puzzle
        self.rows = np.asarray(rows).reshape(-1, len(puzzle.blocks))

    def __len__(self) -> int:
        return len(self.rows)

    @property
    def grids(self) -> np.ndarray:
        """Every solution as a full board-shaped grid, (k, *board shape) --
        built on each access, at ~550 bytes a grid."""
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
