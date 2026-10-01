"""JSON helpers shared by games_io and results_io: a pretty printer that lays
out letter grids one row per line, and the conversion between the letter
rows a grid is stored as on disk and letter arrays / grids of block
indices."""
import itertools
import json
import math
from pathlib import Path

import numpy as np

from constants import EMPTY, OUTSIDE_BOARD


def parse_letter_grid(raw) -> np.ndarray:
    """Row strings (a list of them for a flat board, a list of such lists
    for a pyramid's layers) -> str array of single letters, shaped as
    written, i.e. (depth, width) or (height, depth, width)."""
    def split(item):
        return list(item) if isinstance(item, str) else [split(x) for x in item]
    return np.array(split(raw), dtype=str)


def block_grids_to_rows(grids: np.ndarray, blocks) -> list:
    """Full board-shaped grids of block indices, (k, *board shape), as
    letter rows, one entry per grid: each block's cells become its letter,
    everything else (empty, or outside the board) a space. `blocks` is an
    idx -> Block mapping. One vectorised pass, however many grids."""
    grids = np.asarray(grids)
    offset = -min(EMPTY, OUTSIDE_BOARD)  # grid values start at the sentinels
    lut = np.full(max(blocks) + offset + 1, ord(" "), dtype=np.uint8)
    for idx, block in blocks.items():
        lut[idx + offset] = ord(block.letter)
    letters = lut[grids + offset]
    # to the written orientation: (k, [height,] depth, width)
    written = np.ascontiguousarray(letters.transpose(0, *range(letters.ndim - 1, 0, -1)))
    _, *outer, width = written.shape
    text = written.tobytes().decode("ascii")
    rows = [text[i:i + width] for i in range(0, len(text), width)]
    for n in reversed(outer):  # rows into layers, layers into grids
        rows = [rows[i:i + n] for i in range(0, len(rows), n)]
    return rows


def rows_to_block_grids(entries: list, blocks, shape: tuple) -> np.ndarray:
    """Inverse of block_grids_to_rows for fully solved grids: a list of
    letter-row entries (each shaped like a board of internal `shape`) ->
    one (k, *shape) array of block indices, spaces -> OUTSIDE_BOARD. Raises
    ValueError on an entry of the wrong size or an unknown letter."""
    strings = entries
    for _ in shape:  # a grid nests one level per axis: rows, [layers,] grids
        strings = itertools.chain.from_iterable(strings)
    codes = np.frombuffer("".join(strings).encode("ascii"), dtype=np.uint8)
    if codes.size != len(entries) * math.prod(shape):
        raise ValueError(f"Not every grid has the board's {math.prod(shape)} cells.")
    dtype = np.int8 if max(blocks) < np.iinfo(np.int8).max else np.int16
    unknown = np.iinfo(dtype).min
    lut = np.full(256, unknown, dtype=dtype)
    lut[ord(" ")] = OUTSIDE_BOARD
    for idx, block in blocks.items():
        lut[ord(block.letter)] = idx
    written = lut[codes].reshape(len(entries), *reversed(shape))
    if (written == unknown).any():
        raise ValueError("A grid holds a letter that is no block's.")
    return written.transpose(0, *range(len(shape), 0, -1))


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
