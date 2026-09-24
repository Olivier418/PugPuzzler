import json
from pathlib import Path

from classes import SolveStats, SolveStatsBook
from .jsonio import dump_json
from .solutions_io import single_entry


def write_solve_stats(stats_book: SolveStatsBook, file_path: Path, flat: bool = True) -> None:
    """Mirrors write_solutions: puzzle names are only written for a
    `flat=False` save. The object wrapper stays either way for the sibling
    `options`/`seed` fields."""
    stats = single_entry(stats_book, flat, "SolveStatsBook")
    data = {"options": stats_book.options, "seed": stats_book.seed}
    if flat:
        data |= {"duration": stats.duration, "elapsed": stats.elapsed}
    else:
        data["puzzles"] = [
            {
                "puzzle_name": s.puzzle_name,
                "options": s.options,
                "seed": s.seed,
                "duration": s.duration,
                "elapsed": s.elapsed,
            }
            for s in stats_book.values()
        ]
    dump_json(data, file_path)


def load_solve_stats_book(file_path: Path, flat_name: str, book_name: str = None) -> SolveStatsBook:
    """Inverse of write_solve_stats; a flat file's stats belong to `flat_name`."""
    with open(file_path, "r") as f:
        data = json.load(f)

    book_options = dict(data.get("options", {}))
    book_seed = data.get("seed")

    if "puzzles" in data:
        stats = [
            SolveStats(
                puzzle_name=item["puzzle_name"],
                options=dict(item.get("options", book_options)),
                seed=item.get("seed", book_seed),
                duration=item["duration"],
                elapsed=item["elapsed"],
            )
            for item in data["puzzles"]
        ]
    else:
        stats = [
            SolveStats(
                puzzle_name=flat_name,
                options=book_options,
                seed=book_seed,
                duration=data["duration"],
                elapsed=data["elapsed"],
            )
        ]

    return SolveStatsBook(*stats, book_name=book_name, options=book_options, seed=book_seed)
