from collections import UserDict

from ._utils import _assert_unique, _shared
from .blocks import BlockCollection
from .boards import Board
from .setup import Setup


class PuzzleBook(UserDict):
    def __init__(self, *puzzles, name: str = None):
        if not puzzles:
            raise ValueError(f"A PuzzleBook needs at least one puzzle (book {name!r} is empty).")
        _assert_unique(puzzles, lambda p: p.name, "puzzle name")
        # Every puzzle in a book is built from one shared Setup (that's
        # the whole point -- see Setup's docstring), so the book can
        # expose it directly instead of making callers reach into an
        # arbitrary puzzle's .setup themselves.
        self.setup: Setup = _shared(puzzles, lambda p: p.setup, "Setup")
        self.name = name

        # Same idea as State.source: where this book was loaded from, if
        # anywhere. Populated by serialization.loading.load_game.
        self.source = None

        super().__init__({p.name: p for p in puzzles})

    @property
    def board(self) -> Board:
        return self.setup.board

    @property
    def blocks(self) -> BlockCollection:
        return self.setup.blocks

    def __repr__(self) -> str:
        header = f"Puzzle Book {self.name}" if self.name else "Puzzle Book"
        body = "\n\n".join(p.render(show_leftover=False) for p in self.values())
        return f"{header}\n\n{body}"
