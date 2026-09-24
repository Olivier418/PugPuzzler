from ._utils import _assert_unique
from .puzzlebook import PuzzleBook
from .setup import Setup
from .state import Puzzle


class Game:
    """Container for a game package, holding puzzle books, standalone loose
    puzzles, and board setups.
    """

    def __init__(
        self,
        books: list[PuzzleBook] = None,
        puzzles: list[Puzzle] = None,
        setups: dict[str, Setup] = None,
        name: str = None,
    ):
        books = books or []
        puzzles = puzzles or []

        _assert_unique(books, lambda b: b.name, "puzzlebook name")
        _assert_unique(puzzles, lambda p: p.name, "puzzle name")

        self.name = name
        self.books: dict[str, PuzzleBook] = {b.name: b for b in books}
        self.puzzles: dict[str, Puzzle] = {p.name: p for p in puzzles}
        self.setups: dict[str, Setup] = setups or {}

    def __repr__(self) -> str:
        header = f"Game {self.name}" if self.name else "Game"
        parts = [repr(b) for b in self.books.values()] + [repr(p) for p in self.puzzles.values()]
        return f"{header}\n\n" + "\n\n".join(parts) if parts else header