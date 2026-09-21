from collections import UserDict
from typing import NamedTuple

import numpy as np

from constants import EMPTY
from .state import State


class PuzzleInfo(NamedTuple):
    difficulty: str | None
    nr_empty_spaces: int


class SolveStats(NamedTuple):
    """Metadata about the solver run that produced a Solution -- kept
    separate from Solution itself since the solver is exhaustive: the
    solutions are deterministic, but how long it took / which solver
    options+seed were used is a property of the run, not of the
    solutions. `options` is whatever keyword arguments the solver was
    given besides the seed (currently none; kept so runs of different
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
    """Lean output container referencing a puzzle by ID, plus -- when it
    was produced in this session rather than loaded back from disk -- a
    live reference to the Puzzle it was solved from. A Solution loaded
    from disk has no live puzzle; use `serialization.solution_states` /
    `serialization.solution_puzzle_info` to resolve those from the game
    files instead.

    Producing one (solving + saving) lives in `solving.py`."""

    def __init__(
        self,
        puzzle_name: str,
        grids: list[np.ndarray],
        game_name: str = None,
        book_name: str = None,
        puzzle=None,
    ):
        self.puzzle_name = puzzle_name
        self.grids = grids
        self.game_name = game_name
        self.book_name = book_name
        self.puzzle = puzzle  # live reference only; never written to disk

    @property
    def setup(self):
        return self.puzzle.setup if self.puzzle is not None else None

    def to_states(self, setup=None) -> list[State]:
        """Hydrate result grids into executable State objects, using
        `setup` or else the live puzzle's. Raises without either."""
        setup = setup or self.setup
        if setup is None:
            raise ValueError(
                "No Setup to hydrate with: this Solution has no live puzzle. Pass `setup=` "
                "or use serialization.solution_states."
            )
        states = []
        for grid in self.grids:
            state = State(setup)
            state.grid = setup.to_compact_grid(grid)
            states.append(state)
        return states

    def puzzle_info(self, puzzles=None) -> PuzzleInfo:
        """The source Puzzle's difficulty and empty-cell count. These
        aren't stored on Solution itself -- they're facts about the
        Puzzle, and duplicating them here would let them drift -- so they
        are read off the live `puzzle`, else off `puzzles` (a PuzzleBook,
        or any puzzle_name -> Puzzle mapping). With neither, use
        serialization.solution_puzzle_info, which reads the game's JSON."""
        puzzle = self.puzzle if self.puzzle is not None else (puzzles or {}).get(self.puzzle_name)
        if puzzle is None:
            raise ValueError(
                "No Puzzle to read info from: this Solution has no live puzzle and none was "
                "passed in `puzzles`. Use serialization.solution_puzzle_info."
            )
        return PuzzleInfo(puzzle.difficulty, int((puzzle.grid == EMPTY).sum()))

    def __repr__(self) -> str:
        if self.setup is None:
            return f"Solution to puzzle {self.puzzle_name} ({len(self.grids)} solutions)"
        if not self.grids:
            return f"Solution to puzzle {self.puzzle_name} (no results)"
        return "\n\n".join(
            state.setup.render(state.grid, header=f"Solution {i} to puzzle {self.puzzle_name}")
            for i, state in enumerate(self.to_states(), start=1)
        )


class SolutionBook(UserDict):
    """Container for batch solution results."""
    def __init__(
        self,
        *solutions: Solution,
        game_name: str = None,
        book_name: str = None,
    ):
        self.game_name = game_name
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
        game_name: str = None,
        book_name: str = None,
        options: dict = None,
        seed: int = None,
    ):
        self.game_name = game_name
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
