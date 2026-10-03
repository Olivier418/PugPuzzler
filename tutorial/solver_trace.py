"""A step-by-step view of the solver's branching rule, for tutorial/1_solver.ipynb.

Every node is computed with the kernel's own primitives (choose_item, the
live-placement counters and collectors), so what is shown is exactly what
the search does at that node. walk replays the search node by node, trying
the branches in order and backing out of dead ends, until it finds the
first solution.
"""
import re
from typing import NamedTuple

import numpy as np

from classes import Puzzle
from classes import kernel
from classes.rendering import block_shape_lines
from classes.solver import _start
from constants import EMPTY, UNPLACED

_UNCAPPED = 1 << 30
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


class Step(NamedTuple):
    """One node of the search."""
    kind: str                         # "cell" or "block": what the node branches on
    index: int                        # a compact cell index, or a block index
    cell_counts: np.ndarray           # live placements per compact cell, -1 where filled
    block_counts: dict[int, int]      # live placements per unplaced block
    candidates: list[tuple[int, int]]  # the branches, as (block_idx, placement_idx)

    @property
    def count(self) -> int:
        return len(self.candidates)


def branch_step(puzzle: Puzzle) -> Step:
    """The node `puzzle` is: every item's live count, the item the solver
    branches on (kernel.choose_item, default branch rule) and its
    candidates in the order the search tries them unranked."""
    t = puzzle.setup.kernel_tables
    occ, used = _start(puzzle, t)
    nw = occ.shape[0]

    cell_counts = np.full(t.n_cells, -1, dtype=np.int64)
    for c in np.flatnonzero(puzzle.grid == EMPTY):
        cell_counts[c] = kernel._count_cell(t, c, occ, used, nw, _UNCAPPED)
    block_counts = {
        idx: int(kernel._count_block(t, idx, occ, nw, _UNCAPPED))
        for idx, p in puzzle.chosen_placement_idx.items() if p == UNPLACED
    }

    kind, index = kernel.choose_item(t, occ, used, nw, kernel.BRANCH_BOTH)
    out = np.empty(t.fan, dtype=np.int32)
    collect = kernel.collect_cell if kind == 0 else kernel.collect_block
    gids = out[:collect(t, index, occ, used, nw, out)]
    candidates = [(int(t.pblock[g]), int(g - t.block_start[t.pblock[g]])) for g in gids]
    return Step("cell" if kind == 0 else "block", int(index), cell_counts, block_counts, candidates)


def walk(puzzle: Puzzle) -> tuple[list[tuple[int, Puzzle, Step]], Puzzle | None]:
    """The search from `puzzle` up to its first solution, as the solver
    runs it: every node it visits in order, dead ends included, as
    (depth, puzzle at that node, its Step), and the solved puzzle (None if
    there is no solution). Each node's branches are tried in the order of
    its candidates, the order the kernel tries them unranked."""
    nodes = []

    def visit(node: Puzzle, depth: int) -> Puzzle | None:
        if not (node.grid == EMPTY).any():
            return node
        step = branch_step(node)
        nodes.append((depth, node, step))
        for block_idx, placement_idx in step.candidates:
            child = node.copy()
            child.place_unchecked(block_idx, placement_idx)
            if (solved := visit(child, depth + 1)) is not None:
                return solved
        return None

    return nodes, visit(puzzle, 0)


def _side_by_side(panels: list[list[str]], gap: int = 4, per_row: int = 4) -> str:
    """Multi-line strings next to each other, padded by visible width
    (color codes don't count), `per_row` panels to a row."""
    out = []
    for i in range(0, len(panels), per_row):
        group = panels[i:i + per_row]
        widths = [max(len(_ANSI.sub("", line)) for line in panel) for panel in group]
        height = max(len(panel) for panel in group)
        for r in range(height):
            parts = []
            for panel, width in zip(group, widths):
                line = panel[r] if r < len(panel) else ""
                parts.append(line + " " * (width - len(_ANSI.sub("", line))))
            out.append((" " * gap).join(parts).rstrip())
        out.append("")
    return "\n".join(out).rstrip("\n")


def _item(puzzle: Puzzle, step: Step) -> str:
    """What a node branches on, as words: "cell (x, y)" or "block L"."""
    setup = puzzle.setup
    if step.kind == "block":
        return f"block {setup.blocks[step.index].letter}"
    coords = np.unravel_index(setup.compact_to_flat[step.index], setup.board.cells.shape)
    return f"cell {tuple(int(x) for x in coords)}"


def _placements(n: int) -> str:
    return f"{n} placement" + ("" if n == 1 else "s")


def describe(nodes: list[tuple[int, Puzzle, Step]], solved: Puzzle | None) -> str:
    """The search `walk` returned as a tree indented by depth: per node the
    item it branches on and how many branches, a dead end or forced move on
    an extra-indented line of its own, the solution under the node that
    reaches it, and a closing "..." where the search carries on with
    branches not yet tried."""
    lines, untried = [], None
    for i, (depth, node, step) in enumerate(nodes):
        indent = "  " * depth
        head = f"{indent}{_item(node, step)}"
        # the last node is the solution's parent; every other branch tried is a child node
        reaches_solution = solved is not None and i == len(nodes) - 1
        tried = int(reaches_solution)
        for d, _, _ in nodes[i + 1:]:
            if d <= depth:
                break
            tried += d == depth + 1
        if tried < step.count:
            untried = depth
        if step.count >= 2:
            lines.append(f"{head}, {step.count} branches")
        else:
            what = "dead end" if step.count == 0 else "forced"
            lines += [head, f"{indent}    {what}" + (" --> *solution!*" if reaches_solution else "")]
    if untried is not None:  # under the deepest node with branches left: the search goes on there
        lines.append("  " * (untried + 1) + "...")
    return "\n".join(lines)


def render_cell_counts(puzzle: Puzzle, step: Step) -> str:
    """The board with every empty cell showing how many placements could
    still cover it, and the blocks already placed as plain color."""
    counts = step.cell_counts
    labels = np.array([str(n) if n >= 0 else "" for n in counts], dtype=object)
    cell_width = max(2, len(str(counts.max())) + 1)
    return puzzle.setup.render(puzzle.grid, labels=labels, cell_width=cell_width, letters=False)


def render_block_counts(puzzle: Puzzle, step: Step) -> str:
    """Every unplaced block, drawn in full, with how many placements it
    still has underneath."""
    shapes = {idx: block_shape_lines(puzzle.blocks[idx])[0] for idx in step.block_counts}
    height = max(map(len, shapes.values()))  # captions on one line under the tallest
    panels = [shapes[idx] + [""] * (height - len(shapes[idx])) + ["", _placements(n)]
              for idx, n in step.block_counts.items()]
    return _side_by_side(panels, per_row=len(panels))


def render_branches(puzzle: Puzzle, step: Step) -> str:
    """The item the node branches on, and every branch as the board it
    leads to, numbered in the order the search tries them."""
    setup = puzzle.setup
    if step.kind == "cell":
        lines = [f"Branching on {_item(puzzle, step)}: {_placements(step.count)} cover{'s' if step.count == 1 else ''} it", ""]
    else:
        lines = [f"Branching on {_item(puzzle, step)}: {_placements(step.count)}", ""]

    panels = []
    for i, (block_idx, placement_idx) in enumerate(step.candidates, 1):
        child = puzzle.copy()
        child.place_unchecked(block_idx, placement_idx)
        panels.append([f"branch {i}", ""] + setup.render(child.grid).split("\n"))
    lines.append(_side_by_side(panels))
    return "\n".join(lines)
