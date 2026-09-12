from collections import UserDict

from ._utils import _assert_unique
from .puzzle import Puzzle, PuzzleBook


class SolvedPuzzle:
    """A puzzle bundled with its (already computed) solutions."""
    def __init__(self, puzzle: Puzzle, solutions: list[Puzzle]):
        self.puzzle = puzzle
        self.solutions = solutions


class SolutionBook(UserDict):
    def __init__(self, *solved_puzzles: SolvedPuzzle):
        _assert_unique(solved_puzzles, lambda sp: sp.puzzle.name, "puzzle name")
        super().__init__({sp.puzzle.name: sp for sp in solved_puzzles})

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
            solutions = list(puzzle.solve(mode=mode, seed=seed))  # solve() is expensive: called exactly once
            if disp:
                print(f"Found {len(solutions)} solutions for {puzzle.difficulty} puzzle '{puzzle.name}'")
            solved_puzzles.append(SolvedPuzzle(puzzle, solutions))
        return cls(*solved_puzzles)