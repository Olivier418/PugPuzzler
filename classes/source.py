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

    def relative_dir(self) -> Path:
        """Directory, relative to some root, uniquely identifying where
        this source's own files/runs live -- mirrors its position inside
        games/. A whole book -> 'IQpuzzler/books/main'. A standalone
        puzzle -> 'IQpuzzler/puzzles/main_empty'. One puzzle solved out
        of a book -> nested under its book rather than concatenated into
        one name, so it stays unambiguous to split back apart:
        'IQpuzzler/books/main/some_puzzle'."""
        if self.book_name and self.puzzle_name:
            return Path(self.game_name) / "books" / self.book_name / self.puzzle_name
        if self.book_name:
            return Path(self.game_name) / "books" / self.book_name
        return Path(self.game_name) / "puzzles" / self.puzzle_name

    def for_puzzle(self, puzzle_name: str) -> "Source":
        """The Source for one puzzle inside a book this Source points at."""
        return replace(self, puzzle_name=puzzle_name)