from collections import UserDict
from typing import NamedTuple

import numpy as np

from .setup import Setup
from .state import Puzzle, State


class SolveStats(NamedTuple):
    """Metadata about the solver run that produced a Solution -- kept
    separate from Solution itself since the solver is exhaustive: the
    solutions are deterministic, but how long it took / which solver
    options+seed were used is a property of the run, not of the
    solutions. `options` is whatever keyword arguments the solver was
    given besides the seed (`branch`, `order`; kept so runs of different
    solver variants can be told apart and compared by benchmark.py)."""
    puzzle_name: str
    options: dict
    seed: int | None
    duration: float
    elapsed: list[float]  # per-solution elapsed time, same order/index as the matching Solution.grids


def _book_name(book_name: str | None, puzzle_names) -> str | None:
    """Display/folder name shared by SolutionBook and SolveStatsBook: the
    book's own name, or -- for a book-less container wrapping a single
    puzzle -- that puzzle's name. Never stored separately, so it can't
    drift from book_name."""
    if book_name:
        return book_name
    puzzle_names = list(puzzle_names)
    return puzzle_names[0] if len(puzzle_names) == 1 else None


class Solution:
    """One Puzzle's solutions: the Puzzle itself plus every grid that
    solves it (full board-shaped, see Setup.to_full_grid).

    The Puzzle is always live -- the one solving.py solved, or the one
    serialization.load_solution_run found in games/ -- so everything
    about it (name, difficulty, empty spaces, Setup) is read off it rather
    than copied here, where it could drift.

    Producing one (solving + saving) lives in `solving.py`."""

    def __init__(self, puzzle: Puzzle, grids: list[np.ndarray]):
        self.puzzle = puzzle
        self.grids = grids

    @property
    def puzzle_name(self) -> str:
        return self.puzzle.name

    @property
    def setup(self) -> Setup:
        return self.puzzle.setup

    def to_states(self) -> list[State]:
        """The solved grids as States on the puzzle's Setup."""
        states = []
        for grid in self.grids:
            state = State(self.setup)
            state.grid = self.setup.to_compact_grid(grid)
            states.append(state)
        return states

    def __repr__(self) -> str:
        if not self.grids:
            return f"Solution to puzzle {self.puzzle_name} (no results)"
        return "\n\n".join(
            self.setup.render(state.grid, header=f"Solution {i} to puzzle {self.puzzle_name}")
            for i, state in enumerate(self.to_states(), start=1)
        )


class SolutionBook(UserDict):
    """Container for batch solution results, keyed by puzzle name."""
    def __init__(self, *solutions: Solution, book_name: str = None):
        self.book_name = book_name
        super().__init__({sol.puzzle_name: sol for sol in solutions})

    @property
    def name(self) -> str | None:
        return _book_name(self.book_name, self)

    def __repr__(self) -> str:
        header = f"Solution Book {self.name}" if self.name else "Solution Book"
        body = "\n\n".join(repr(sol) for sol in self.values())
        return f"{header}\n\n{body}" if body else header


class SolveStatsBook(UserDict):
    """Container for batch solve stats, mirroring SolutionBook's shape."""
    def __init__(
        self,
        *stats: SolveStats,
        book_name: str = None,
        options: dict = None,
        seed: int = None,
    ):
        self.book_name = book_name
        self.options = dict(options or {})
        self.seed = seed
        super().__init__({s.puzzle_name: s for s in stats})

    @property
    def name(self) -> str | None:
        return _book_name(self.book_name, self)

    def __repr__(self) -> str:
        header = f"Solve Stats {self.name}" if self.name else "Solve Stats"
        lines = [
            f"{s.puzzle_name}: options={s.options} seed={s.seed} duration={s.duration:.3f}s "
            f"({len(s.elapsed)} solutions)"
            for s in self.values()
        ]
        body = "\n".join(lines)
        return f"{header}\n{body}" if body else header
