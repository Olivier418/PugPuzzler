from collections import UserDict
from pathlib import Path
import time
from typing import NamedTuple
import numpy as np

from constants import EMPTY
from .puzzle import State
from .source import Source


class Result(NamedTuple):
    grid: np.ndarray
    elapsed: float


class PuzzleInfo(NamedTuple):
    difficulty: str | None
    nr_empty_spaces: int


def _next_solutions_path(source: Source | None, solutions_root: str | Path) -> Path:
    """Where to auto-save a Solution/SolutionBook: mirrors `source`'s
    game/puzzle-or-book directory under `solutions_root` (no extra
    per-run subfolder, so the tree stays symmetric with games/), using
    the next free '{stem}_solutions_{idx}.json' name so repeated solves
    of the same puzzle/book accumulate instead of overwriting each other."""
    if source is None:
        raise ValueError(
            "Can't auto-save: no `source` on this Puzzle/PuzzleBook (it "
            "wasn't loaded via serialization.load_game). Pass `path=` "
            "explicitly, or `save=False` to skip saving."
        )
    directory = Path(solutions_root) / source.relative_dir()
    idx = 1
    while (directory / f"{source.stem}_solutions_{idx}.json").exists():
        idx += 1
    return directory / f"{source.stem}_solutions_{idx}.json"


class Solution:
    """Lean output container referencing a puzzle by ID, plus -- when it
    was produced in this session rather than loaded back from disk -- a
    live reference to the Setup it was solved against, so to_states()
    doesn't need to reload anything from disk."""

    def __init__(
        self,
        puzzle_name: str,
        results: list[Result],
        duration: float,
        game_name: str = None,
        book_name: str = None,
        mode: int = 2,
        seed: int = None,
        setup=None,
    ):
        self.puzzle_name = puzzle_name
        self.results = results
        self.duration = duration
        self.game_name = game_name
        self.book_name = book_name
        self.mode = mode
        self.seed = seed
        self.setup = setup  # live reference only; never written to disk

    @classmethod
    def from_puzzle(
        cls,
        puzzle,
        mode: int = 2,
        seed: int = None,
        disp: bool = False,
        save: bool = True,
        path: str | Path = None,
        solutions_root: str | Path = "solutions",
    ) -> "Solution":
        """Solve a single Puzzle and, by default, save the result to a
        folder mirroring where the Puzzle itself was loaded from, under
        `solutions_root` (e.g. games/IQpuzzler/puzzles/main_empty ->
        solutions/IQpuzzler/puzzles/main_empty/<timestamp>/solutions.json).

        Pass `save=False` to just solve, or `path=` to save somewhere
        specific instead of the mirrored default.
        """
        start = time.perf_counter()
        results = [
            Result(state.grid.copy(), time.perf_counter() - start)
            for state in puzzle.solve(mode=mode, seed=seed, disp=disp)
        ]
        duration = time.perf_counter() - start

        source = puzzle.source
        solution = cls(
            puzzle_name=puzzle.name,
            results=results,
            duration=duration,
            game_name=source.game_name if source else None,
            book_name=source.book_name if source else None,
            mode=mode,
            seed=seed,
            setup=puzzle.setup,
        )

        if save:
            from serialization import save_solutions  # lazy: avoids a classes<->serialization import cycle

            target = Path(path) if path is not None else _next_solutions_path(source, solutions_root)
            book = SolutionBook(
                solution,
                game_name=solution.game_name,
                book_name=solution.book_name,
                name=puzzle.name,
            )
            save_solutions(book, target)

        return solution

    def to_states(self, setup=None, games_root: str | Path = "games") -> list[State]:
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
        for res in self.results:
            state = State(setup)
            state.grid = res.grid.copy()
            states.append(state)
        return states

    def puzzle_info(self, puzzles=None, games_root: str | Path = "games"):
        """Look up this solution's originating Puzzle's difficulty and
        empty-cell count. `difficulty`/`nr_empty_spaces` aren't stored on
        Solution itself -- they're facts about the Puzzle, not the
        Solution, and duplicating them here would let them drift from
        the source of truth -- so this resolves them on demand instead,
        same spirit as to_states() resolving a Setup on demand.

        Pass `puzzles` (a PuzzleBook, or any puzzle_name -> Puzzle
        mapping) when one is already in memory, e.g. right after
        SolutionBook.from_puzzlebook(book) -- pass `book` itself. With no
        `puzzles`, falls back to reading the puzzle's JSON directly via
        the canonical game/book/puzzle IDs; unlike to_states()'s Setup
        fallback, this never needs the expensive placement computation.
        """
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
        mode: int = 2,
        seed: int = None,
        name: str = None,
    ):
        self.game_name = game_name
        self.book_name = book_name
        self.mode = mode
        self.seed = seed
        self.name = name or book_name
        super().__init__({sol.puzzle_name: sol for sol in solutions})

    @classmethod
    def from_puzzlebook(
        cls,
        puzzlebook,
        game_name: str = None,
        mode: int = 2,
        seed: int = None,
        disp: bool = False,
        name: str = None,
        save: bool = True,
        path: str | Path = None,
        solutions_root: str | Path = "solutions",
    ) -> "SolutionBook":
        """Solve every puzzle in a PuzzleBook and, by default, save the
        combined result to a folder mirroring where the book itself was
        loaded from (see Solution.from_puzzle for the mirroring rule).

        Delegates per-puzzle solving to Solution.from_puzzle so the two
        entry points can't drift apart; only the batching and the single
        combined save are specific to this method.
        """
        source = puzzlebook.source
        book_name = puzzlebook.name
        game_name = game_name or (source.game_name if source else None)
        solutions = []

        for puzzle in puzzlebook.values():
            solution = Solution.from_puzzle(puzzle, mode=mode, seed=seed, save=False)
            # Individual puzzles carry their own Source, but this
            # SolutionBook is filed under the book's own game/book name --
            # keep every Solution in it consistent with that.
            solution.game_name = game_name
            solution.book_name = book_name
            solutions.append(solution)
            if disp:
                print(f"Found {len(solution.results)} solutions for puzzle '{puzzle.name}'")

        result = cls(
            *solutions,
            game_name=game_name,
            book_name=book_name,
            mode=mode,
            seed=seed,
            name=name,
        )

        if save:
            from serialization import save_solutions  # lazy: avoids a classes<->serialization import cycle

            target = Path(path) if path is not None else _next_solutions_path(source, solutions_root)
            save_solutions(result, target)

        return result

    def __repr__(self) -> str:
        header = f"Solution Book {self.name}" if self.name else "Solution Book"
        body = "\n\n".join(repr(sol) for sol in self.values())
        return f"{header}\n\n{body}" if body else header