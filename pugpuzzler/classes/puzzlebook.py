from collections import UserDict

from ._utils import _assert_unique
from .setup import Setup


class PuzzleBook(UserDict):
    def __init__(self, *puzzles, name: str = None):
        if not puzzles:
            raise ValueError(f"A PuzzleBook needs at least one puzzle (book {name!r} is empty).")
        _assert_unique(puzzles, lambda p: p.name, "puzzle name")
        self.setup: Setup = puzzles[0].setup  # shared by all of them, see Setup
        if any(p.setup is not self.setup for p in puzzles):
            raise ValueError(f"The puzzles of book {name!r} aren't all on the same Setup.")
        self.name = name

        self.source = None  # a Source, set by the serialization.games_io loaders

        super().__init__({p.name: p for p in puzzles})

    def __repr__(self) -> str:
        header = f"Puzzle Book {self.name}" if self.name else "Puzzle Book"
        body = "\n\n".join(p.render(show_leftover=False) for p in self.values())
        return f"{header}\n\n{body}"
