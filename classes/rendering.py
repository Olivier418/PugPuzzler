"""Terminal rendering for boards/blocks -- kept separate from `Setup`
(classes/setup.py) since this is a display concern, not part of the
board/blocks/placements model itself. `Setup.render` is a thin
delegator to `render` below; everything here operates on plain
`Board`/`BlockCollection`/grid arguments so it has no dependency on
`Setup` itself.
"""
import numpy as np
from colorama import Style

from constants import EMPTY, OUTSIDE_BOARD
from .blocks import Block, BlockCollection
from .boards import Board


def _colored_cell(block: Block) -> str:
    return f"{block.terminal_color}{block.letter} {Style.RESET_ALL}"


def grid_lines(board: Board, blocks: BlockCollection, grid: np.ndarray) -> list[str]:
    """Render any grid shaped like `board` -- a State's own grid, a
    solved grid from Solver, or otherwise -- into a list of
    terminal-ready row strings (one per output row, already colored)."""
    # Unpack (width, depth, height)
    shape = board.cells.shape
    width, depth, height = (shape[0], shape[1], 1) if len(shape) == 2 else shape
    reshaped_grid = grid.reshape(width, depth, height)

    layer_strings = []

    # Iterate over height (3rd axis / index 2)
    for h in range(height):
        layer = reshaped_grid[:, :, h]  # (width, depth) horizontal slice
        padded = np.pad(layer, pad_width=2, constant_values=OUTSIDE_BOARD)
        lines = []

        if height > 1:
            visual_width = (width + 2) * 2
            lines.append(f"layer {h}".ljust(visual_width))

        # Iterate over depth rows (r) and width cols (c)
        for r in range(1, depth + 3):
            row_chars = []
            for c in range(1, width + 3):
                cell = padded[c, r]

                if cell == EMPTY:
                    row_chars.append('  ')
                elif cell >= 0:
                    row_chars.append(_colored_cell(blocks[cell]))
                else:  # cell == OUTSIDE_BOARD
                    n, s = padded[c, r-1] != OUTSIDE_BOARD, padded[c, r+1] != OUTSIDE_BOARD
                    w, e = padded[c-1, r] != OUTSIDE_BOARD, padded[c+1, r] != OUTSIDE_BOARD
                    nw, ne = padded[c-1, r-1] != OUTSIDE_BOARD, padded[c+1, r-1] != OUTSIDE_BOARD
                    sw, se = padded[c-1, r+1] != OUTSIDE_BOARD, padded[c+1, r+1] != OUTSIDE_BOARD

                    # a hole is a bracketed square; a notch open to the top/bottom is a U-shaped
                    # slit in the edge: ┗┛ / ┏┓ in the notch itself, ┓┏ / ┛┗ in the wall cell
                    # on its open side, so the board edge runs continuously around it
                    if n and s and w and e: char = '[]'
                    elif w and e and s:     char = '┗┛'
                    elif w and e and n:     char = '┏┓'
                    elif w and e:           char = '┃┃'  # middle of a taller hole
                    elif n and s and w:     char = '┃ '
                    elif n and s and e:     char = ' ┃'
                    elif n and w: char = '┏━'
                    elif n and e: char = '━┓'
                    elif s and w: char = '┗━'
                    elif s and e: char = '━┛'
                    elif n or s:  char = '━━'
                    elif w:       char = '┃ '
                    elif e:       char = ' ┃'
                    elif nw and ne: char = '┛┗'
                    elif sw and se: char = '┓┏'
                    elif nw:      char = '┛ '
                    elif ne:      char = ' ┗'
                    elif sw:      char = '┓ '
                    elif se:      char = ' ┏'
                    else:         char = '  '

                    row_chars.append(char)

            lines.append("".join(row_chars))
        layer_strings.append(lines)

    return ['  '.join(row_tuple) for row_tuple in zip(*layer_strings)]


def block_shape_lines(block: Block) -> tuple[list[str], int]:
    """Render a single block's own shape (from its `coords`, not any
    board) as a small standalone diagram: same 2-char-per-cell,
    colored-letter style as `grid_lines`, but a tight bounding box with
    no board border/corner-drawing. Returns the lines plus their width
    in cells (not raw string length, since colorama escape codes make
    raw length meaningless for alignment)."""
    coords = block.coords - block.coords.min(axis=0)
    extent = coords.max(axis=0) + 1
    width, depth, height = (list(extent) + [1, 1])[:3]

    grid = np.zeros((width, depth, height), dtype=bool)
    padded_idx = [coords[:, d] if d < block.ndim else np.zeros(len(coords), dtype=int) for d in range(3)]
    grid[tuple(padded_idx)] = True

    layer_strings = []
    for h in range(height):
        layer = grid[:, :, h]
        lines = []
        if height > 1:
            lines.append(f"layer {h}".ljust(width * 2))
        for r in range(depth):
            row_chars = []
            for c in range(width):
                if layer[c, r]:
                    row_chars.append(_colored_cell(block))
                else:
                    row_chars.append('  ')
            lines.append(''.join(row_chars))
        layer_strings.append(lines)

    combined_width = width * height + max(height - 1, 0)  # layers joined with a 1-cell gap
    return ['  '.join(row_tuple) for row_tuple in zip(*layer_strings)], combined_width


def legend_lines(blocks: BlockCollection, block_idcs) -> list[str]:
    """Arrange the shape diagrams of several blocks (e.g. a State's
    unplaced blocks) side by side in one row, used to lay unplaced blocks
    out underneath the board."""
    entries = [block_shape_lines(blocks[idx]) for idx in sorted(block_idcs)]
    if not entries:
        return []

    row_height = max(len(lines) for lines, _ in entries)
    columns = [lines + ['  ' * width] * (row_height - len(lines)) for lines, width in entries]
    return ['  '.join(col[r] for col in columns) for r in range(row_height)]


def render(
    board: Board,
    blocks: BlockCollection,
    grid: np.ndarray,
    header: str = None,
    leftover_idcs=None,
) -> str:
    """Build the full text representation used by every `__repr__` that
    displays a board: an optional header, then the board, and -- when
    `leftover_idcs` is given -- shape diagrams of those blocks laid out
    underneath the board."""
    lines = grid_lines(board, blocks, grid)

    if leftover_idcs:
        legend = legend_lines(blocks, leftover_idcs)
        if legend:
            lines = lines + [''] + legend

    body = "\n".join(lines)
    return f"{header}\n\n{body}" if header else body
