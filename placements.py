"""How many valid placements each piece has on each board of a Game."""
import re

from colorama import Back, Fore, Style

from classes import Game
from classes.rendering import block_shape_lines

_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def placement_counts(game: Game) -> dict[str, dict[int, int]]:
    """{board_key: {block_idx: number of valid placements of that block}}."""
    return {
        board_key: {idx: len(cells) for idx, cells in setup.placement_cells.items()}
        for board_key, setup in game.setups.items()
    }


def _visible_len(s: str) -> int:
    return len(_ANSI.sub("", s))


def _pad(s: str, width: int) -> str:
    return s + " " * (width - _visible_len(s))


def print_placement_table(game: Game) -> dict[str, dict[int, int]]:
    """Print the counts as a table, the board(s) where each piece has
    the most placements highlighted in white (rows = pieces drawn as in the terminal
    renderer, columns = boards) and return them."""
    counts = placement_counts(game)
    boards = list(counts)
    blocks = next(iter(game.setups.values())).blocks

    shapes = {idx: block_shape_lines(block)[0] for idx, block in blocks.items()}
    shape_w = max(_visible_len(line) for lines in shapes.values() for line in lines)
    col_w = [max(len(b), *(len(str(counts[b][idx])) for idx in blocks)) for b in boards]

    best = {idx: max(counts[b][idx] for b in boards) for idx in blocks}

    def num(b, idx, w):
        text = str(counts[b][idx]).rjust(w)
        return f"{Back.WHITE}{Fore.BLACK}{text}{Style.RESET_ALL}" if counts[b][idx] == best[idx] else text

    def rule(l, m, r):
        return l + m.join("─" * (w + 2) for w in [shape_w, *col_w]) + r

    def row(cells):
        return "│ " + " │ ".join(cells) + " │"

    out = [rule("┌", "┬", "┐"), row([_pad("piece", shape_w)] + [b.rjust(w) for b, w in zip(boards, col_w)])]
    for idx in blocks:
        out.append(rule("├", "┼", "┤"))
        lines = shapes[idx]
        mid = len(lines) // 2
        for i, line in enumerate(lines):
            nums = [num(b, idx, w) if i == mid else " " * w for b, w in zip(boards, col_w)]
            out.append(row([_pad(line, shape_w)] + nums))
    out.append(rule("└", "┴", "┘"))
    print("\n".join(out))
    return counts


if __name__ == "__main__":
    from serialization import load_game

    print_placement_table(load_game("games/IQpuzzler"))
