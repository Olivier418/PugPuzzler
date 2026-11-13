from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Source:
    """Where a Puzzle or PuzzleBook was loaded from, as names relative to a
    games root (not a path), so they can be re-rooted under solutions/ etc.
    Set by the serialization.games_io loaders and save_puzzlebook; None on
    anything built by hand."""

    game_name: str
    book_name: str | None = None   # set for puzzles/books loaded from books/
    puzzle_name: str | None = None   # set for a single puzzle (standalone, or one entry of a book)

    def relative_dir(self) -> Path:
        """This source's position inside games/, as a relative directory:
        'IQpuzzler/books/main_puzzles' (a book), 'IQpuzzler/puzzles/empty_main'
        (a standalone puzzle), 'IQpuzzler/books/main_puzzles/50' (one puzzle
        of a book, nested so it splits back apart unambiguously)."""
        if self.book_name and self.puzzle_name:
            return Path(self.game_name) / "books" / self.book_name / self.puzzle_name
        if self.book_name:
            return Path(self.game_name) / "books" / self.book_name
        return Path(self.game_name) / "puzzles" / self.puzzle_name