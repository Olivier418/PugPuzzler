from collections import UserDict

from ._utils import _assert_unique, _shared
from .boards import Board
from .blocks import BlockCollection
from .puzzle import Puzzle, PuzzleBook, PuzzleSetup


import time
from typing import NamedTuple


class Result(NamedTuple):
    solution: Puzzle
    elapsed: float  # seconds since solving started on the parent puzzle


class SolvedPuzzle:
    """A puzzle bundled with its (already computed) solutions, each
    tagged with how long the solver took to reach it, plus how long the
    solver ran on this puzzle in total. `duration` is not necessarily
    the elapsed time of the last result -- the solver may keep running
    (e.g. exhausting the search space or hitting a timeout) after the
    last solution was already found."""
    def __init__(self, puzzle: Puzzle, results: list[Result], duration: float):
        self.puzzle = puzzle
        self.results = results
        self.duration = duration


class SolutionBook(UserDict):
    def __init__(self, *solved_puzzles: SolvedPuzzle, mode: int, seed: int = None):
        _assert_unique(solved_puzzles, lambda sp: sp.puzzle.name, "puzzle name")
        self.setup: PuzzleSetup = _shared(solved_puzzles, lambda sp: sp.puzzle.setup, "PuzzleSetup")
        self.mode = mode
        self.seed = seed
        super().__init__({sp.puzzle.name: sp for sp in solved_puzzles})

    @property
    def board(self) -> Board:
        return self.setup.board

    @property
    def blocks(self) -> BlockCollection:
        return self.setup.blocks

    @classmethod
    def from_puzzlebook(
        cls, puzzlebook: PuzzleBook, mode: int = 2, seed: int = None, disp: bool = False
    ) -> "SolutionBook":
        """Solve every puzzle in puzzlebook and return the results as a
        SolutionBook. Pure in-memory computation -- no file I/O happens
        here. Use serialization.save_solutions separately if you want to
        persist the result, and serialization.load_solutions to later
        recover a SolutionBook without paying the solve cost again.
        """
        solved_puzzles = []
        for puzzle in puzzlebook.values():
            start = time.perf_counter()
            results = []
            for solution in puzzle.solve(mode=mode, seed=seed):
                results.append(Result(solution, time.perf_counter() - start))
            # captured after the solve generator is fully exhausted, so it
            # reflects the whole time the solver spent on this puzzle --
            # not just the moment the last solution was yielded
            duration = time.perf_counter() - start
            if disp:
                print(f"Found {len(results)} solutions for {puzzle.difficulty} puzzle '{puzzle.name}'")
            solved_puzzles.append(SolvedPuzzle(puzzle, results, duration))
        return cls(*solved_puzzles, mode=mode, seed=seed)