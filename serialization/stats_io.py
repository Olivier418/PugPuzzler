import json
from pathlib import Path

from classes import SolveStats, SolveStatsBook
from .paths import game_book_from_run_dir, puzzle_name_from_run_dir, single_entry


def write_solve_stats(stats_book: SolveStatsBook, file_path: Path, flat: bool = True) -> None:
    """Mirrors write_solutions: game_name/book_name/puzzle_name are
    never written, since the folder already determines them. The object
    wrapper is kept either way since it always carries sibling
    `options`/`seed` fields alongside the per-puzzle data."""
    stats = single_entry(stats_book, flat, "SolveStatsBook")
    if flat:
        data = {
            "options": stats_book.options,
            "seed": stats_book.seed,
            "duration": stats.duration,
            "elapsed": stats.elapsed,
        }
    else:
        data = {
            "options": stats_book.options,
            "seed": stats_book.seed,
            "puzzles": [
                {
                    "puzzle_name": s.puzzle_name,
                    "options": s.options,
                    "seed": s.seed,
                    "duration": s.duration,
                    "elapsed": s.elapsed,
                }
                for s in stats_book.values()
            ],
        }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


# Solver flags that older stats.json files recorded as top-level keys before
# they became the generic "options" dict; folded back into it on load so old
# runs (and benchmark folders) still load, and still tell configs apart.
_LEGACY_OPTION_KEYS = ("partition_pruning", "partition_branching", "symmetry_branching")


def _read_options(data: dict, default: dict | None = None) -> dict:
    """The solver options recorded in a stats.json (or one puzzle's entry
    in it): the "options" dict if present, else any legacy flag keys, else
    `default` (the book-level value, for a per-puzzle entry that has none
    of its own)."""
    if "options" in data:
        return dict(data["options"])
    legacy = {k: data[k] for k in _LEGACY_OPTION_KEYS if k in data}
    if legacy:
        return legacy
    return dict(default or {})


def load_solve_stats_book(path: str | Path) -> SolveStatsBook:
    """Load a stats.json (or the run folder holding one)."""
    path = Path(path)
    file_path = path / "stats.json" if path.is_dir() else path

    with open(file_path, "r") as f:
        data = json.load(f)

    game_name, book_name = game_book_from_run_dir(file_path)
    book_options = _read_options(data)
    book_seed = data.get("seed")

    if "puzzles" in data:
        stats = [
            SolveStats(
                puzzle_name=item["puzzle_name"],
                options=_read_options(item, default=book_options),
                seed=item.get("seed", book_seed),
                duration=item["duration"],
                elapsed=item["elapsed"],
            )
            for item in data["puzzles"]
        ]
    else:
        stats = [
            SolveStats(
                puzzle_name=puzzle_name_from_run_dir(file_path),
                options=book_options,
                seed=book_seed,
                duration=data["duration"],
                elapsed=data["elapsed"],
            )
        ]

    return SolveStatsBook(
        *stats, game_name=game_name, book_name=book_name, options=book_options, seed=book_seed
    )
