import json
from datetime import datetime
from pathlib import Path

import numpy as np

from classes import SolutionBook, Solution
from classes.solutions import Result


def _write_solutions(solution_book: SolutionBook, file_path: Path) -> None:
    data = {
        "name": solution_book.name,
        "game_name": solution_book.game_name,
        "book_name": solution_book.book_name,
        "mode": solution_book.mode,
        "seed": solution_book.seed,
        "puzzles": [
            {
                "puzzle_name": sol.puzzle_name,
                "mode": sol.mode,
                "seed": sol.seed,
                "duration": sol.duration,
                "results": [
                    {
                        "grid": res.grid.tolist(),
                        "elapsed": res.elapsed,
                    }
                    for res in sol.results
                ],
            }
            for sol in solution_book.values()
        ],
    }
    with open(file_path, "w") as f:
        json.dump(data, f, indent=2)


def save_solutions(solution_book: SolutionBook, file_path: str | Path) -> Path:
    file_path = Path(file_path)

    if file_path.suffix == ".json":
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _write_solutions(solution_book, file_path)
        return file_path

    name = solution_book.name or "solutions"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = file_path / name / timestamp
    folder.mkdir(parents=True, exist_ok=True)
    _write_solutions(solution_book, folder / "solutions.json")
    return folder


def _read_solutions(path: str | Path) -> tuple[list[Solution], dict]:
    """Parse a solutions.json into its Solutions plus the book-level
    metadata (game_name, book_name, mode, seed, name) -- the one place
    that reads the file, shared by load_solutionbook and load_solution
    so neither has to go through the other's container type."""
    path = Path(path)
    file_path = path / "solutions.json" if path.is_dir() else path

    with open(file_path, "r") as f:
        data = json.load(f)

    game_name = data.get("game_name")
    book_name = data.get("book_name")
    book_mode = data.get("mode", 2)
    book_seed = data.get("seed")

    solutions = [
        Solution(
            puzzle_name=item["puzzle_name"],
            results=[
                Result(
                    grid=np.array(res["grid"], dtype=int),
                    elapsed=res["elapsed"],
                )
                for res in item["results"]
            ],
            duration=item["duration"],
            game_name=game_name,
            book_name=book_name,
            mode=item.get("mode", book_mode),
            seed=item.get("seed", book_seed),
        )
        for item in data["puzzles"]
    ]

    meta = {
        "game_name": game_name,
        "book_name": book_name,
        "mode": book_mode,
        "seed": book_seed,
        "name": data.get("name"),
    }
    return solutions, meta


def load_solutionbook(path: str | Path) -> SolutionBook:
    solutions, meta = _read_solutions(path)
    return SolutionBook(*solutions, **meta)


def load_solution(path: str | Path) -> Solution:
    """Load a solutions file saved from a single puzzle (see
    Solution.from_puzzle) and return the bare Solution, instead of
    making the caller index a one-entry SolutionBook themselves.

    Raises if the file holds more than one puzzle's solutions -- use
    load_solutionbook() for those instead.
    """
    solutions, _ = _read_solutions(path)
    if len(solutions) != 1:
        names = ", ".join(sol.puzzle_name for sol in solutions)
        raise ValueError(
            f"Expected a single-puzzle solutions file, found {len(solutions)} puzzles "
            f"({names}) in {path}. Use load_solutionbook() instead."
        )
    return solutions[0]
