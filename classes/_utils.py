from collections import Counter


def _assert_unique(items, key_fn, label: str) -> None:
    """Raise ValueError if key_fn(item) is not unique across items."""
    counts = Counter(key_fn(item) for item in items)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {label}: {duplicates}")