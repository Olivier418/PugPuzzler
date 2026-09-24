"""Where runs live on disk, and how to recover names from a run folder.

Every writer saves under `<root>/<game>/(books/<book>[/<puzzle>] | puzzles/<puzzle>)/<prefix>_<idx>`
(mirroring `Source.relative_dir()`), so a run never stores which puzzles it
solved -- the path points at them in games/, and they're recovered from it
here. Leaf module: depends only on `classes.source`."""
from pathlib import Path

from classes.source import Source


def next_free_idx_dir(base: Path, prefix: str = "result") -> Path:
    """base/{prefix}_1, base/{prefix}_2, ... -- the next not-yet-existing
    per-run subfolder under base. Shared by every place that auto-saves
    a run (solutions, benchmarks) so re-running the same puzzle/book
    always gets its own folder instead of overwriting the last one."""
    idx = 1
    while (base / f"{prefix}_{idx}").exists():
        idx += 1
    return base / f"{prefix}_{idx}"


def next_run_dir(source: Source | None, root: str | Path, prefix: str = "result") -> Path:
    """Where to auto-save a run: mirrors `source`'s game/puzzle-or-book
    directory under `root`, then the next free `<prefix>_<idx>` folder --
    re-solving the same puzzle/book is expected to reproduce the same
    solutions, but repeated runs still each get their own folder."""
    if source is None:
        raise ValueError(
            "Can't save a run of a Puzzle/PuzzleBook that isn't in games/ (no `source`): a run "
            "refers to its puzzles there. Give the book a home first with "
            "serialization.save_puzzlebook."
        )
    return next_free_idx_dir(Path(root) / source.relative_dir(), prefix)


def _anchor(path: Path) -> tuple[tuple[str, ...], int]:
    """The path's parts and the index of its nearest `books`/`puzzles`
    component -- the one Source.relative_dir() puts right below the game."""
    parts = Path(path).parts
    for i in range(len(parts) - 2, 0, -1):
        if parts[i] == "puzzles" or (parts[i] == "books" and i + 2 < len(parts)):
            return parts, i
    raise ValueError(
        f"Can't tell which puzzles the run in {path} belongs to: its path doesn't follow "
        "<game>/(books/<book>|puzzles)/..."
    )


def game_book_from_run_dir(path: Path) -> tuple[str, str | None]:
    """(game_name, book_name) of a run folder -- the inverse of
    Source.relative_dir(), book_name None for a standalone puzzle. A
    benchmark's config/seed folders nested further down are fine."""
    parts, i = _anchor(path)
    return parts[i - 1], parts[i + 1] if parts[i] == "books" else None


def puzzle_name_from_run_dir(path: Path) -> str:
    """The puzzle a single-puzzle run belongs to: the folder right below
    <game>/books/<book>/ or <game>/puzzles/. (For a whole-book run that's
    the run folder itself, so only ask this of a single-puzzle one.)"""
    parts, i = _anchor(path)
    return parts[i + 2] if parts[i] == "books" else parts[i + 1]
