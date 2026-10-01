"""Shared test helpers. Tests read games/ and tests/data/ and write only to
temporary folders, never solutions/, benchmarks/ or bounds/."""
import json
from pathlib import Path

from constants import GAMES_DIR
from serialization import load_game

ROOT = Path(__file__).resolve().parent.parent
GAMES_ROOT = ROOT / GAMES_DIR

PENTOMINOES = {
    "F": [".XX", "XX.", ".X."], "I": ["XXXXX"],        "L": ["X.", "X.", "X.", "XX"],
    "N": [".X", ".X", "XX", "X."], "P": ["XX", "XX", "X."], "T": ["XXX", ".X.", ".X."],
    "U": ["X.X", "XXX"],        "V": ["X..", "X..", "XXX"], "W": ["X..", "XX.", ".XX"],
    "X": [".X.", "XXX", ".X."], "Y": [".X", "XX", ".X", ".X"], "Z": ["XX.", ".X.", ".XX"],
}


def load(game_name: str):
    return load_game(GAMES_ROOT / game_name)


def rows(puzzle, **options) -> list[tuple]:
    """Every solution the solver finds, as sorted row tuples."""
    return sorted(map(tuple, puzzle.solve_rows(**options)))


def write_pentomino_game(games_root: Path) -> Path:
    """A game folder "Pentominoes" under `games_root`: the 12 pentominoes on
    a 3x20 board, with its empty puzzle `empty_3x20`. The board has 8
    solutions, a published count (4x15: 1472, 5x12: 4040, 6x10: 9356 --
    minutes each, worth a run by hand after touching the kernel)."""
    folder = games_root / "Pentominoes"
    (folder / "puzzles").mkdir(parents=True)
    blocks = {
        letter: {
            "letter": letter,
            "positions": [[int(c == "X") for c in row] for row in shape],
            "terminal": {"bg": "WHITE", "fg": "BLACK"},
        }
        for letter, shape in PENTOMINOES.items()
    }
    (folder / "blocks.json").write_text(json.dumps(blocks))
    (folder / "boards.json").write_text(json.dumps({"3x20": {"type": "RegularBoard", "width": 20, "depth": 3}}))
    (folder / "puzzles" / "empty_3x20.json").write_text(json.dumps({"board": "3x20"}))
    return folder
