from .puzzle import Puzzle
from .puzzlebook import PuzzleBook
from .setup import Setup


class Game:
    """One games/<name>/ folder: its Setups (board key -> Setup), its books
    and its standalone puzzles (both by name, which are file stems, so
    unique)."""

    def __init__(self, name: str, setups: dict[str, Setup], books: list[PuzzleBook], puzzles: list[Puzzle]):
        self.name = name
        self.setups = setups
        self.books: dict[str, PuzzleBook] = {b.name: b for b in books}
        self.puzzles: dict[str, Puzzle] = {p.name: p for p in puzzles}

    def board_key(self, setup: Setup) -> str:
        """The key of `setup` in `self.setups` (matched by identity)."""
        key = next((key for key, s in self.setups.items() if s is setup), None)
        if key is None:
            raise ValueError(f"That Setup isn't one of game {self.name!r}'s (it has {sorted(self.setups)}).")
        return key

    def __repr__(self) -> str:
        header = f"Game {self.name}"
        parts = [repr(b) for b in self.books.values()] + [repr(p) for p in self.puzzles.values()]
        return f"{header}\n\n" + "\n\n".join(parts) if parts else header
