from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Source:
    """Where a Puzzle or PuzzleBook was loaded from, expressed as
    coordinates relative to a games root -- not a live path -- so the same
    coordinates can be re-rooted onto a solutions directory later.

    Set once by the loader (see serialization.loading.load_game) and left
    as None for anything built by hand in memory.
    """

    game_name: str
    book_name: str | None = None   # set for puzzles/books loaded from books/
    puzzle_name: str | None = None   # set for a single puzzle (standalone, or one entry of a book)

    @property
    def stem(self) -> str:
        """Base filename this source's solutions should be saved under.
        A whole book -> its own name ('main'). A standalone puzzle -> its
        own name ('main_empty'). One puzzle solved out of a book -> both,
        so it's still unambiguous which book it came from ('main_somepuzzle')."""
        if self.puzzle_name and self.book_name:
            return f"{self.book_name}_{self.puzzle_name}"
        return self.puzzle_name or self.book_name

    def relative_dir(self) -> Path:
        """Directory, relative to some root, mirroring this source's
        position inside its game -- e.g. 'IQpuzzler/books' or
        'IQpuzzler/puzzles'. Mirrors where the *file* lives in games/
        (games/IQpuzzler/puzzles/main_empty.json), not a per-puzzle
        subfolder -- the filename (see `stem`) carries the specific name."""
        return Path(self.game_name) / ("books" if self.book_name else "puzzles")

    def for_puzzle(self, puzzle_name: str) -> "Source":
        """The Source for one puzzle inside a book this Source points at."""
        return replace(self, puzzle_name=puzzle_name)