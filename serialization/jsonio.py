"""JSON helpers shared by the puzzle/solution readers and writers: a pretty
printer that lays out letter grids one row per line, and the conversion
between a letter grid (rows of single-character cells) and the row strings
it is stored as on disk."""
import json
from pathlib import Path

import numpy as np


def parse_letter_grid(raw) -> np.ndarray:
    """Row strings (a list of them for a flat board, a list of such lists
    for a pyramid's layers) -> str array of single letters, shaped as
    written, i.e. (depth, width) or (height, depth, width)."""
    def split(item):
        return list(item) if isinstance(item, str) else [split(x) for x in item]
    return np.array(split(raw), dtype=str)


def letter_grid_to_rows(arr: np.ndarray) -> list:
    """Inverse of parse_letter_grid for an *internal* (width, depth[, height])
    letter array: transposed back to the written orientation, then each row
    joined into one string."""
    def join(a):
        return ["".join(row) for row in a] if a.ndim == 2 else [join(x) for x in a]
    return join(np.asarray(arr).T)


def _format(obj, indent: int, col: int) -> str:
    """`indent`: indentation of the line this value's brackets close on;
    `col`: the column its first character sits at (differs from `indent`
    after a dict key), which a row block aligns its later rows to."""
    pad = " " * indent
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = []
        for key, value in obj.items():
            prefix = f'{json.dumps(key)}: '
            items.append(f'{pad}  {prefix}{_format(value, indent + 2, indent + 2 + len(prefix))}')
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, str) for x in obj):
            # a block of rows: one per line, aligned under the first
            return "[" + (",\n" + " " * (col + 1)).join(json.dumps(x) for x in obj) + "]"
        if not any(isinstance(x, (list, dict)) for x in obj):
            return json.dumps(obj)
        items = [f"{pad}  {_format(x, indent + 2, indent + 2)}" for x in obj]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(obj)


def dump_json(obj, path: str | Path) -> None:
    """Write `obj` as indented JSON, except that a list of strings (a
    letter grid's rows) is printed one row per line, aligned under the
    first, instead of one element per line."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(_format(obj, 0, 0) + "\n")
