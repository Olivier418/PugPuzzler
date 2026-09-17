from collections import Counter
from pathlib import Path


def _next_free_idx_dir(base: Path, prefix: str = "result") -> Path:
    """base/{prefix}_1, base/{prefix}_2, ... -- the next not-yet-existing
    per-run subfolder under base. Shared by every place that auto-saves
    a run (solutions, solve stats, benchmarks) so re-running the same
    puzzle/book always gets its own folder instead of overwriting the
    last one."""
    idx = 1
    while (base / f"{prefix}_{idx}").exists():
        idx += 1
    return base / f"{prefix}_{idx}"


def _assert_unique(items, key_fn, label: str) -> None:
    """Raise ValueError if key_fn(item) is not unique across items."""
    counts = Counter(key_fn(item) for item in items)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {label}: {duplicates}")


def _shared(items, key_fn, label: str):
    """Return the single value of key_fn shared (by identity) across every
    item, or None if items is empty. Raises if items disagree -- e.g. two
    puzzles in the same book built from different PuzzleSetups, which
    should never happen but is worth catching rather than silently
    picking one."""
    if not items:
        return None
    first = key_fn(items[0])
    if not all(key_fn(item) is first for item in items):
        raise ValueError(f"All items must share the same {label}.")
    return first