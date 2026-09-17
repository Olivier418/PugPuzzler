from collections import UserDict
from pathlib import Path
import time
from typing import NamedTuple
import numpy as np

from constants import EMPTY, SOLUTION_DIR, GAMES_DIR
from ._utils import _next_free_idx_dir
from .puzzle import State
from .source import Source


class PuzzleInfo(NamedTuple):
    difficulty: str | None
    nr_empty_spaces: int


class SolveStats(NamedTuple):
    """Metadata about the solver run that produced a Solution -- kept
    separate from Solution itself since the solver is exhaustive: the
    solutions are deterministic, but how long it took / what mode+seed
    were used is a property of the run, not of the solutions."""
    puzzle_name: str
    mode: int
    seed: int | None
    duration: float
    elapsed: list[float]  # per-solution elapsed time, same order/index as the matching Solution.grids


def _next_run_dir(source: Source | None, solutions_root: str | Path) -> Path:
    """Where to auto-save a Solution/SolutionBook + its SolveStats:
    mirrors `source`'s game/puzzle-or-book directory under
    `solutions_root`, then a per-run subfolder named with the next free
    index -- re-solving the same puzzle/book is expected to reproduce
    the same solutions, but repeated runs (e.g. while chasing a solver
    bug) still each get their own folder instead of overwriting."""
    if source is None:
        raise ValueError(
            "Can't auto-save: no `source` on this Puzzle/PuzzleBook (it "
            "wasn't loaded via serialization.load_game). Pass `path=` "
            "explicitly, or `save=False` to skip saving."
        )
    base = Path(solutions_root) / source.relative_dir()
    return _next_free_idx_dir(base)


class Solution:
    """Lean output container referencing a puzzle by ID, plus -- when it
    was produced in this session rather than loaded back from disk -- a
    live reference to the Puzzle it was solved from, so to_states() and
    puzzle_info() don't need to reload anything from disk."""

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

    @classmethod
    def from_puzzle(
        cls,
        puzzle,
        mode: int = 2,
        seed: int = None,
        disp: bool = False,
        save: bool = True,
        path: str | Path = None,
        solutions_root: str | Path = SOLUTION_DIR,
    ) -> tuple["Solution", SolveStats]:
        """Solve a single Puzzle and, by default, save the results and
        the run's SolveStats to a folder mirroring where the Puzzle
        itself was loaded from, under `solutions_root` (e.g.
        games/IQpuzzler/puzzles/main_empty ->
        solutions/IQpuzzler/puzzles/main_empty/<idx>/{solutions,stats}.json).

        Pass `save=False` to just solve, or `path=` to save somewhere
        specific instead of the mirrored default.
        """
        start = time.perf_counter()
        grids = []
        elapsed = []
        for state in puzzle.solve(mode=mode, seed=seed, disp=disp):
            grids.append(state.grid.copy())
            elapsed.append(time.perf_counter() - start)
        duration = time.perf_counter() - start

        source = puzzle.source
        solution = cls(
            puzzle_name=puzzle.name,
            grids=grids,
            game_name=source.game_name if source else None,
            book_name=source.book_name if source else None,
            puzzle=puzzle,
        )
        stats = SolveStats(
            puzzle_name=puzzle.name,
            mode=mode,
            seed=seed,
            duration=duration,
            elapsed=elapsed,
        )

        if save:
            from serialization import save_solution_run  # lazy: avoids a classes<->serialization import cycle

            target = Path(path) if path is not None else _next_run_dir(source, solutions_root)
            solution_book = SolutionBook(solution, game_name=solution.game_name, book_name=solution.book_name)
            stats_book = SolveStatsBook(stats, game_name=solution.game_name, book_name=solution.book_name, mode=mode, seed=seed)
            save_solution_run(solution_book, stats_book, target)

        return solution, stats

    def to_states(self, setup=None, games_root: str | Path = GAMES_DIR) -> list[State]:
        """Hydrate result grids into executable State objects.

        Uses the live `setup` reference when there is one (set on any
        Solution produced by from_puzzle/from_puzzlebook in this
        session); otherwise falls back to resolving a Setup from the
        canonical game/book/puzzle IDs, for a Solution loaded back from
        a bare solutions.json with no Puzzle in memory.
        """
        setup = setup or self.setup
        if setup is None:
            if not self.game_name:
                raise ValueError("Cannot resolve Setup without game_name.")
            from serialization import load_setup_for_puzzle
            setup = load_setup_for_puzzle(
                game_name=self.game_name,
                book_name=self.book_name,
                puzzle_name=self.puzzle_name,
                games_root=games_root,
            )

        states = []
        for grid in self.grids:
            state = State(setup)
            state.grid = grid.copy()
            states.append(state)
        return states

    def puzzle_info(self, puzzles=None, games_root: str | Path = GAMES_DIR):
        """Look up this solution's originating Puzzle's difficulty and
        empty-cell count. `difficulty`/`nr_empty_spaces` aren't stored on
        Solution itself -- they're facts about the Puzzle, not the
        Solution, and duplicating them here would let them drift from
        the source of truth -- so this resolves them on demand instead,
        same spirit as to_states() resolving a Setup on demand.

        Uses the live `puzzle` reference when there is one (set by
        from_puzzle/from_puzzlebook in this session). Otherwise, pass
        `puzzles` (a PuzzleBook, or any puzzle_name -> Puzzle mapping)
        when one is in memory -- e.g. a freshly `load_game`'d book
        matching a Solution reloaded from a bare solutions.json. With
        neither, falls back to reading the puzzle's JSON directly via
        the canonical game/book/puzzle IDs; unlike to_states()'s Setup
        fallback, this never needs the expensive placement computation.
        """
        if self.puzzle is not None:
            return PuzzleInfo(self.puzzle.difficulty, int((self.puzzle.grid == EMPTY).sum()))

        if puzzles is not None:
            puzzle = puzzles[self.puzzle_name]
            return PuzzleInfo(puzzle.difficulty, int((puzzle.grid == EMPTY).sum()))

        if not self.game_name:
            raise ValueError("Cannot resolve puzzle info without game_name.")
        from serialization import load_puzzle_info_for_puzzle
        return load_puzzle_info_for_puzzle(
            game_name=self.game_name,
            book_name=self.book_name,
            puzzle_name=self.puzzle_name,
            games_root=games_root,
        )

    def __repr__(self) -> str:
        states = self.to_states()
        if not states:
            return f"Solution to puzzle {self.puzzle_name} (no results)"
        parts = [
            state.setup.render(state.grid, header=f"Solution {i} to puzzle {self.puzzle_name}")
            for i, state in enumerate(states, start=1)
        ]
        return "\n\n".join(parts)


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
        """Display/folder name: the book's own name, or -- for a book-less
        SolutionBook wrapping a single puzzle's solutions -- that puzzle's
        name. Never stored separately, so it can't drift from book_name."""
        if self.book_name:
            return self.book_name
        if len(self) == 1:
            return next(iter(self))
        return None

    @classmethod
    def from_puzzlebook(
        cls,
        puzzlebook,
        game_name: str = None,
        mode: int = 2,
        seed: int = None,
        disp: bool = False,
        save: bool = True,
        path: str | Path = None,
        solutions_root: str | Path = SOLUTION_DIR,
    ) -> tuple["SolutionBook", "SolveStatsBook"]:
        """Solve every puzzle in a PuzzleBook and, by default, save the
        combined solutions and solve stats to a folder mirroring where
        the book itself was loaded from (see Solution.from_puzzle for
        the mirroring rule).

        Delegates per-puzzle solving to Solution.from_puzzle so the two
        entry points can't drift apart; only the batching and the single
        combined save are specific to this method.
        """
        source = puzzlebook.source
        book_name = puzzlebook.name
        game_name = game_name or (source.game_name if source else None)
        solutions = []
        stats_list = []

        for puzzle in puzzlebook.values():
            solution, stats = Solution.from_puzzle(puzzle, mode=mode, seed=seed, save=False)
            # Individual puzzles carry their own Source, but this
            # SolutionBook is filed under the book's own game/book name --
            # keep every Solution in it consistent with that.
            solution.game_name = game_name
            solution.book_name = book_name
            solutions.append(solution)
            stats_list.append(stats)
            if disp:
                print(f"Found {len(solution.grids)} solutions for puzzle '{puzzle.name}'")

        result = cls(
            *solutions,
            game_name=game_name,
            book_name=book_name,
        )
        stats_book = SolveStatsBook(
            *stats_list,
            game_name=game_name,
            book_name=book_name,
            mode=mode,
            seed=seed,
        )

        if save:
            from serialization import save_solution_run  # lazy: avoids a classes<->serialization import cycle

            target = Path(path) if path is not None else _next_run_dir(source, solutions_root)
            save_solution_run(result, stats_book, target, flat=False)

        return result, stats_book

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
        mode: int = 2,
        seed: int = None,
    ):
        self.game_name = game_name
        self.book_name = book_name
        self.mode = mode
        self.seed = seed
        super().__init__({s.puzzle_name: s for s in stats})

    @property
    def name(self) -> str | None:
        """Same rule as SolutionBook.name."""
        if self.book_name:
            return self.book_name
        if len(self) == 1:
            return next(iter(self))
        return None

    def __repr__(self) -> str:
        header = f"Solve Stats {self.name}" if self.name else "Solve Stats"
        lines = [
            f"{s.puzzle_name}: mode={s.mode} seed={s.seed} duration={s.duration:.3f}s "
            f"({len(s.elapsed)} solutions)"
            for s in self.values()
        ]
        body = "\n".join(lines)
        return f"{header}\n{body}" if body else header
