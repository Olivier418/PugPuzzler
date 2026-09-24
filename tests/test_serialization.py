"""Serialization tests: letter-grid JSON helpers, the run-folder round
trip and reloading a benchmark. Writes only into a temporary folder, never
solutions/ or benchmarks/."""
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmark import config_key, config_label, load_benchmark
from classes import PuzzleBook, SolveStats, SolveStatsBook
from constants import GAMES_DIR, UNPLACED
from serialization import (
    dump_json, letter_grid_to_rows, load_puzzles, load_run, parse_letter_grid, save_puzzlebook, save_run,
)
from serialization.stats_io import write_solve_stats
from solving import single_run_books, solve_puzzle, solve_puzzlebook
from tests._helpers import load


class TestLetterGrid(unittest.TestCase):
    def test_round_trip_flat_and_layered(self):
        for rows in (["AB ", "CDE"], [["AB", "CD"], ["E ", "  "]]):
            arr = parse_letter_grid(rows)
            self.assertEqual(letter_grid_to_rows(arr.T), rows)

    def test_dump_json_round_trips(self):
        data = {"k": [["AB", "CD"]], "n": [1, 2]}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x.json"
            dump_json(data, path)
            self.assertEqual(json.loads(path.read_text()), data)


def assert_same_run(test, loaded, original):
    """A loaded (PuzzleBook, SolutionBook, SolveStatsBook) run holds the same
    puzzles, solutions and stats as an in-memory (SolutionBook, SolveStatsBook)."""
    (loaded_puzzles, loaded_sols, loaded_stats), (sols, stats) = loaded, original
    test.assertEqual(list(loaded_puzzles), list(sols))
    test.assertEqual(list(loaded_sols), list(sols))
    for name, sol in sols.items():
        got = loaded_sols[name]
        test.assertIs(got.puzzle, loaded_puzzles[name])
        test.assertEqual(got.puzzle.difficulty, sol.puzzle.difficulty)
        test.assertTrue(np.array_equal(got.puzzle.grid, sol.puzzle.grid))
        test.assertEqual(len(got.grids), len(sol.grids))
        for a, b in zip(got.grids, sol.grids):
            test.assertTrue(np.array_equal(a, b))
        test.assertEqual(loaded_stats[name], stats[name])


def hand_made(puzzle, name):
    """`puzzle` with one pre-placed piece taken off, as a new puzzle that
    exists nowhere in games/."""
    made = puzzle.copy(rename=name)
    made.source = None
    made.remove(next(i for i, p in made.chosen_placement_idx.items() if p != UNPLACED))
    return made


def copy_of_games(tmp: str, game_name: str = "IQpuzzler") -> Path:
    """A games root under `tmp` holding a copy of one game, for tests that
    add a book to it (save_puzzlebook) without touching the real games/."""
    games_root = Path(tmp) / "games"
    shutil.copytree(GAMES_DIR / game_name, games_root / game_name)
    return games_root


class TestSolutionRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.game = load("IQpuzzler")

    def test_single_puzzle_round_trip(self):
        for book_name, name in (("main_puzzles", "50"), ("pyramid_puzzles", "85")):
            with self.subTest(book=book_name, puzzle=name):
                puzzle = self.game.books[book_name][name]
                run = single_run_books(*solve_puzzle(puzzle))
                self.assertTrue(run[0][name].grids)
                with tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp) / "IQpuzzler" / "books" / book_name / name / "result_1"
                    save_run(*run, folder)
                    self.assertEqual(sorted(f.name for f in folder.iterdir()), ["solutions.json", "stats.json"])
                    self.assertRegex((folder / "solutions.json").read_text(), r'^[\[\]",\sA-Za-z]*$')
                    loaded = load_run(folder)  # puzzles from games/
                    assert_same_run(self, loaded, run)
                    self.assertEqual(list(loaded[0]), [name])  # only the puzzle that was solved

    def test_whole_book_round_trip(self):
        game = load("IQpuzzler")
        source = game.books["main_puzzles"]
        book = PuzzleBook(source["50"].copy(), source["51"].copy(), name="two")
        with tempfile.TemporaryDirectory() as tmp:
            games_root = copy_of_games(tmp)
            save_puzzlebook(book, game, games_root=games_root)
            folder = Path(tmp) / "IQpuzzler" / "books" / "two" / "result_1"
            run = solve_puzzlebook(book, save_solutions=True, save_stats=True, path=folder)
            assert_same_run(self, load_run(folder, games_root=games_root), run)

    def test_new_book_gets_a_home_before_its_run_is_saved(self):
        game = load("IQpuzzler")
        book = PuzzleBook(hand_made(game.books["main_puzzles"]["50"], "A"), name="made_here")
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):  # nowhere in games/ yet: nothing to refer to
                solve_puzzlebook(book, save_solutions=True, save_stats=True, solutions_root=tmp)

            games_root = copy_of_games(tmp)
            json_file = save_puzzlebook(book, game, games_root=games_root)
            self.assertIs(game.books["made_here"], book)
            reread = load_puzzles(json_file, game.setups)
            self.assertTrue(np.array_equal(reread[0].grid, book["A"].grid))
            with self.assertRaises(FileExistsError):
                save_puzzlebook(book, game, games_root=games_root)

            run = solve_puzzlebook(book, save_solutions=True, save_stats=True, solutions_root=tmp)
            folder = Path(tmp) / "IQpuzzler" / "books" / "made_here" / "result_1"
            assert_same_run(self, load_run(folder, games_root=games_root), run)

    def test_saving_only_one_half(self):
        puzzle = self.game.books["main_puzzles"]["50"]
        for save_solutions, save_stats, files in (
            (True, False, ["solutions.json"]),
            (False, True, ["stats.json"]),
            (False, False, []),
        ):
            with self.subTest(save_solutions=save_solutions, save_stats=save_stats):
                with tempfile.TemporaryDirectory() as tmp:
                    folder = Path(tmp) / "IQpuzzler" / "books" / "main_puzzles" / "50" / "result_1"
                    solution, stats = solve_puzzle(
                        puzzle, save_solutions=save_solutions, save_stats=save_stats, path=folder)
                    if not files:
                        self.assertFalse(folder.exists())  # `path` alone saves nothing
                        continue
                    self.assertEqual(sorted(f.name for f in folder.iterdir()), files)
                    loaded_puzzles, loaded_sols, loaded_stats = load_run(folder)
                    self.assertEqual(list(loaded_puzzles), ["50"])  # the puzzles are always there
                    self.assertTrue(np.array_equal(loaded_puzzles["50"].grid, puzzle.grid))
                    self.assertEqual(loaded_sols is None, not save_solutions)
                    self.assertEqual(loaded_stats is None, not save_stats)
                    if loaded_sols is not None:
                        self.assertEqual(len(loaded_sols["50"].grids), len(solution.grids))
                    if loaded_stats is not None:
                        self.assertEqual(loaded_stats["50"], stats)


class TestBenchmarkLoad(unittest.TestCase):
    def test_load_groups_trials_by_config_in_seed_order(self):
        configs = ({"order": "counts"}, {"order": False})
        seeds = (0, 1, 2, 10)  # 10 sorts after 2 numerically, not as text
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "IQpuzzler" / "puzzles" / "empty_main" / "benchmark_1"
            written = {}
            for config in configs:
                config_folder = folder / config_label(config)
                config_folder.mkdir(parents=True)
                for seed in seeds:
                    stats = SolveStats("empty_main", dict(config), seed, 1.0 + seed, [0.1 * seed, 0.5])
                    stats_book = SolveStatsBook(stats, options=config, seed=seed)
                    write_solve_stats(stats_book, config_folder / f"stats{seed}.json")
                    written.setdefault(config_key(config), []).append(stats)
            loaded = load_benchmark(folder)
            self.assertEqual(loaded.keys(), written.keys())
            for key, trials in written.items():
                self.assertEqual(loaded[key], trials)


if __name__ == "__main__":
    unittest.main()
