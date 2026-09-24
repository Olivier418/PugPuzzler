"""Where runs live on disk, and how to recover names from a run folder.

Every writer saves under `<root>/<game>/(books/<book>[/<puzzle>] | puzzles/<puzzle>)/<prefix>_<idx>`
(mirroring `Source.relative_dir()`), so the game/book/puzzle names never
have to be stored in the files themselves -- they're recovered from the
path here. Leaf module: depends only on `classes.source`."""
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


def next_run_dir(source: Source | None, root: str | Path) -> Path:
    """Where to auto-save a solve: mirrors `source`'s game/puzzle-or-book
    directory under `root`, then the next free `result_<idx>` folder --
    re-solving the same puzzle/book is expected to reproduce the same
    solutions, but repeated runs still each get their own folder."""
    if source is None:
        raise ValueError(
            "Can't auto-save: no `source` on this Puzzle/PuzzleBook (it "
            "wasn't loaded via serialization.load_game). Pass `path=` "
            "explicitly, or drop `save=True`."
        )
    return next_free_idx_dir(Path(root) / source.relative_dir())


def puzzle_name_from_run_dir(file_path: Path) -> str:
    """Recover a flattened file's puzzle_name from its folder, per the
    convention every writer here uses: .../<puzzle_name>/<idx>/<file>."""
    return file_path.parent.parent.name


def game_book_from_run_dir(file_path: Path) -> tuple[str | None, str | None]:
    """Best-effort recovery of (game_name, book_name) from a run folder
    following the <game>/(books/<book>[/<puzzle>]|puzzles/<puzzle>)/<idx>
    convention -- the inverse of Source.relative_dir(). Returns
    (None, None) for a folder that doesn't follow it (e.g. a benchmark's
    nested config/seed layout), where callers already have puzzle_name
    explicit in the JSON and don't need this."""
    parts = file_path.parts
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "puzzles" and i > 0:
            return parts[i - 1], None
        if parts[i] == "books" and i > 0 and i + 1 < len(parts):
            return parts[i - 1], parts[i + 1]
    return None, None


def single_entry(book, flat: bool, label: str):
    """For a `flat` save: the one entry of `book`, raising if there isn't
    exactly one. Returns None when not flat."""
    if not flat:
        return None
    if len(book) != 1:
        raise ValueError(
            f"flat=True requires exactly one puzzle in the {label}, got {len(book)}. "
            "Use flat=False for a multi-puzzle save."
        )
    return next(iter(book.values()))
