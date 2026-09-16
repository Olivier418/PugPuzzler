# Instructions
- **Tone:** Concise and direct.
- **Output Format:** State the solution/fix first in 1–2 sentences. Skip introductory setups, polite preambles, and post-explanations unless requested.
- **Code:** Provide full updated snippets directly. Do not summarize changes in prose if the code speaks for itself. If you change anything about the structure of the code, make sure that you update it in the codebase map below. 

# Codebase map

Core model, layered bottom-up:
- `classes/lattice.py` — `Lattice`: geometry of a board/block's axes (orthogonal vs. offset, e.g. triangular/pyramid). Derives gram2 / unit_vectors / neighbor_structure. Leaf, no deps.
- `classes/blocks.py` — `Block` (shape + color/letter + its own Lattice), `BlockCollection` (UserDict, int idx -> Block, unique letters).
- `classes/boards.py` — `Board` (cells: bool ndarray + Lattice), `RegularBoard`, `PyramidBoard` subclasses.
- `classes/puzzle.py` — the big one:
  - `Setup`: board + blocks + **all valid placements** (expensive, computed once, shared by reference across every Puzzle/Book built on it). `placements[block_idx]` = flat cell-index array per valid orientation/position. `render(grid, header=None, leftover_idcs=None)` is a thin facade delegating to `classes/rendering.py` -- the shared basis for every `__repr__` below.
  - `State`: mutable solve state (grid + chosen_placement_idx) over a shared `Setup`. `place`/`remove` (checked) vs `place_unchecked`/`remove_unchecked` (used by Solver, no overlap/UNPLACED checks). `.solve()` delegates to `Solver.py`. `render(show_leftover=True)` (via `_print_header`, overridable) builds header + grid + shape diagrams of any still-unplaced blocks; `__repr__` calls it with the default, so `print(state)` just works. `show_leftover=False` is used when a container prints many puzzles at once and the per-puzzle legend would just be noise.
  - `Puzzle(State)`: named, has difficulty, optionally pre-filled from a letter grid (`_initialize_grid`, transposes JSON (depth,width) -> internal (width,depth)). Overrides `_print_header` to `"Puzzle {name}"`.
  - `PuzzleBook(UserDict)`: named collection of Puzzles sharing one Setup. `__repr__` expands every contained Puzzle in full, with `show_leftover=False` (a puzzle printed on its own still shows its unplaced blocks; one printed as part of a book doesn't).
- `classes/game.py` — `Game`: top-level container of `books` (dict name->PuzzleBook), `puzzles` (standalone), `setups` (dict board_key->Setup). One Game per `games/<GameName>/` folder. `__repr__` expands every book and standalone puzzle in full.
- `classes/source.py` — `Source` (frozen dataclass): game/book/puzzle name triple recording where a Puzzle/Book was loaded from, so solutions can be auto-saved to a mirrored path. Set by `serialization.loading.load_game`, `None` for hand-built objects.
- `classes/solutions.py`:
  - `Result` (NamedTuple: grid, elapsed) — one solved grid + time-to-find.
  - `PuzzleInfo` (NamedTuple: difficulty, nr_empty_spaces) — the source Puzzle's, not the Solution's own; never stored on `Solution`, only returned on demand (see below) so it can't drift from the Puzzle.
  - `Solution`: puzzle_name + list[Result] + duration + game/book_name + mode/seed + optional live `setup` ref (not persisted). `.from_puzzle()` solves+optionally auto-saves. `.to_states()` hydrates Results back into live `State` objects (uses live setup if present, else `load_setup_for_puzzle`). `.puzzle_info(puzzles=None)` resolves the source Puzzle's difficulty/nr_empty_spaces -- from `puzzles` (a PuzzleBook/dict) if given, else `load_puzzle_info_for_puzzle` (no Setup needed). `__repr__` hydrates via `to_states()` and renders each result as "Solution {i} to puzzle {name}" + grid (via `Setup.render`).
  - `SolutionBook(UserDict)`: puzzle_name -> Solution, plus book-level game/book_name/mode/seed. `.from_puzzlebook()` batch-solves via `Solution.from_puzzle`. `__repr__` expands every contained Solution in full.
- `classes/rendering.py` — terminal display, split out of `Setup` since it's a display concern, not part of the board/blocks/placements model: `grid_lines(board, blocks, grid)` (ASCII/color board rendering as a list of row strings), `block_shape_lines(block)` (a single block's own shape, from `block.coords`, as a small standalone diagram -- returns lines + width in cells, since colorama escape codes make raw string length meaningless for alignment), `legend_lines(blocks, block_idcs, max_width_cells=None)` (lays out several blocks' shape diagrams side by side, wrapping onto further rows once `max_width_cells` is exceeded -- used for a State's unplaced blocks), `render(board, blocks, grid, header=None, leftover_idcs=None)` (header + grid + leftover-block legend underneath, composed into one string). Operates on plain `Board`/`BlockCollection`/grid args, no dependency on `Setup`.
- `classes/_utils.py` — `_assert_unique`, `_shared` (dedup/consistency helpers used across the above).
- `classes/__init__.py` — the public re-export surface; import from here, not submodules, in new code.

Serialization (`serialization/`):
- `loading.py`: `load_blocks`/`load_boards` (parse `blocks.json`/`boards.json`), `load_game(dir)` (auto-discovers `books/*.json` + `puzzles/*.json` under a `games/<Name>/` dir, builds a full `Game`), `load_setup_for_puzzle` (cold-start Setup rebuild for `Solution.to_states` fallback), `load_puzzle_info_for_puzzle` (cold-start difficulty/nr_empty_spaces for `Solution.puzzle_info` fallback -- reads the puzzle's JSON directly, no Setup/placements needed), `grid_to_letter_rows`.
- `saving.py`: `save_solutions` (SolutionBook -> json, either exact path or `<root>/<name>/<timestamp>/solutions.json`), `load_solutionbook`/`load_solution` (inverse; `load_solution` requires exactly one puzzle in the file).
- On-disk layout: `games/<Game>/{blocks.json, boards.json, books/*.json, puzzles/*.json}`; solutions mirror it under `solutions/<Game>/...`; benchmark runs go under `benchmarks/<Game>/<puzzle>/<timestamp>/mode{N}/test{seed}.json`.

Solving: `Solver.py` / `SolverV2.py` (older/newer backtracking search over a `State`'s placements; modes 0/1/2 = no-pruning/pruning/branching), `compact.py`, `partitioning.py` — solver internals, out of scope for structural review per user.

Plotting/benchmarking (`plotting/`): `plot_difficultyspace.py`, `plot_solve_timeline.py` are current -- both take an optional `puzzles` (PuzzleBook/dict) arg and read difficulty/nr_empty_spaces via `Solution.puzzle_info(puzzles)`, not off `Solution` directly. `benchmark.py` (`run_benchmark`/`load_benchmark`/`plot_benchmark`, runs solves in a subprocess with a wall-clock cap `T`) is **STALE**: still uses the old `Solution` API (`sp.puzzle.name`, `sp.puzzle.difficulty`, `sp.puzzle.grid`, and `_run_single_test` calls `G.solve(...)` on a `Game` and constructs `Solution(G, results, duration)` positionally), which predates the `Solution(puzzle_name, results, ...)` refactor in `classes/solutions.py`. Will break if run as-is; check before trusting/using.

`main.py` is a scratch/scrapyard of commented-out usage examples, not a real entrypoint — check what's actually uncommented before assuming it reflects current usage patterns.

`constants.py`: grid sentinel values (EMPTY/OUTSIDE_BOARD/UNPLACED) + color tables for difficulty/mode used only in plotting.

Testing: `test_compact.py` tests `compact.py` (solver internals). `testing.py` is empty/unused.