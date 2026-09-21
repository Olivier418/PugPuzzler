# Adding symmetry to the cell-MRV solver -- findings and starting points

Written at the end of the refactor that made cell-MRV the solver's only branching strategy and deleted block-MRV, partitioning and the old `symmetry_branching`. Everything below is meant to let a later session start from here without re-deriving it. Numbers are single runs on the development machine, not careful benchmarks.

## 1. Where things stand

**Solver (`Solver.py`)** -- exhaustive exact cover, no flags, only `solve(seed=None)`.
- Flat state: `_build_flat_tables` lays every block's placements end to end (`_P`, rows padded with the dummy cell `n_cells`; block position `p` owns global ids `_lo[p]:_lo[p+1]`; `_block_of`, `_cell_lists[c]`, lazily cached `_conflicts_of(gid)`).
- A node is `(live, bcount, cell_count)`: bool array over global ids, per-block live counts (`_DONE` once placed), per-cell live counts. `_root_node()` builds the first; `_place(live, bcount, cell_count, pos, gid)` derives each child in a few array ops (returns `None` if some block is left with no placement).
- `rec` branches on the open cell with the fewest live placements (`_cell_branch_candidates`) and tries every placement covering it, ordered lexicographically by the sorted live counts of each candidate's cells. Ties: per-solve random priorities from the seed (`_priority` per placement, `_cell_priority` per cell; identity when unseeded).
- Solutions are yielded as `game.copy()` snapshots.

**Symmetry geometry that is still in the repo (unchanged, tested):**
- `Lattice.point_group` -- all integer matrices `M` with `M.T @ gram2 @ M == gram2` (8 for square, 48 for the pyramid lattice; says nothing about a particular board).
- `Setup.region_symmetries(mask)` -- every symmetry of an *arbitrary* sub-region of the board's cells, as compact-cell-index permutations that are the identity outside the mask (always includes the identity). Fast path via `Setup._symmetry_tables` (padded flattened positions of every cell under every matrix, so a translation is a scalar add and "lexicographically smallest cell" is a `min`); cost is a handful of numpy ops on `(K, m)` arrays, K = point-group size. `Setup.board_symmetries` = the same on all cells.
- `test_symmetry.py` still contains `test_region_symmetries_*`, `open_pocket_puzzle`, `quadrant_puzzle`, `random_tiling_puzzle` and `reference_solutions` (an independent exact-cover DFS) -- reuse them.

**Where the removed symmetry solver code lives.** The deleted design notes are `git show 343f69e^:SYMMETRY_SOLVER_NOTES.md` (read them: they explain the per-node region sweep, the cost work, and the correctness argument). The last *committed* solver with symmetry is commit `343f69e` (also `244be59`), where it is "mode 4" in `Solver.py`: `_region_symmetries` (memoized on the region mask, capped at 200k entries), `_orbits`, `_apply_symmetry`, `_placement_image`, `_build_placement_lookup`. The three-flag version that immediately preceded this refactor (`partition_pruning` / `partition_branching` / `symmetry_branching`) was never committed; mode 4 is the same idea.

## 2. Why the old scheme worked, and the rule to keep

The old solver branched on one *block* B. Let `R` be the still-open region and `G` its symmetry group (`region_symmetries(R)`; each element maps `R` onto itself and is the identity elsewhere). For `sigma` in `G`:
- `sigma` maps every solution of `R` to another solution of `R`;
- it maps a placement of B inside `R` to another placement of B inside `R` (guaranteed to exist: `Setup._valid_orientations` / `_compute_placement_indices` enumerate every isometric copy of every block).

So the set of B's candidate placements is `G`-invariant, and every solution contains exactly one of them. The solutions therefore split into classes by B's placement, `sigma` maps the class of `p` bijectively onto the class of `sigma(p)`, and it is enough to search one representative `p` per orbit and *derive* the other classes by transforming the representative's solutions. No dedupe is needed and none of the derived solutions is ever found twice.

**The rule: skipping-and-deriving is safe exactly when the branching set `C` satisfies both**
1. `C` is invariant under the symmetries being used (each maps `C` onto itself);
2. every solution contains **exactly one** member of `C`.

Placements of one block: both hold for the whole group. Placements covering one cell `c`: (2) holds, (1) only for symmetries that fix `c`.

## 3. Why plain cell-branching loses the symmetry

Branch set `S_c` = all placements (any block) covering the chosen cell `c`.
- `sigma` maps a placement covering `c` to one covering `sigma(c)`. If `sigma(c) != c` the image is *not* one of this node's branches, so it says nothing about which branches are equivalent.
- Worse, a solution `s` has some placement `p` at `c` and some `q` at `sigma(c)`. `sigma(s)` has `sigma(q)` at `c`, so it lives in branch `sigma(q)` of the *same* node -- and which branch depends on `q`, a choice made deeper in the subtree. Deriving `sigma(s)` from branch `p` would duplicate what branch `sigma(q)` finds on its own.
- Taking *all* the most-constrained cells (a symmetric set: `cell_count` is `G`-invariant, so corners A,B,C,D of a square are all equally scarce) and branching on placements covering any of them makes the set `G`-invariant but breaks (2): every solution covers A and B and C and D, so it belongs to several branches.
- The dedupe-free variant "explore `p`, derive all images, then forbid every image of `p` in later branches" needs dedupe for solutions containing both `p` and an image of `p`. Ruled out.

## 4. Options that fit the rule (no dedupe), roughly in order of promise

**A. Hybrid per node: symmetric region -> branch on a block, otherwise branch on a cell.** The flat state supports a block branch directly: block position `pos`'s candidates are `gids = np.flatnonzero(live[_lo[pos]:_lo[pos+1]]) + _lo[pos]`, the fewest-placements block is `argmin(bcount)`, and `_place` already handles it. Full group, no dedupe. New code: the block branch step, orbit computation over gids, derived solutions.

**B. Cell-branching with the stabilizer only.** Keep branching on cell `c`; use only images with `image[c] == c` (the stabilizer). `S_c` is invariant under those and each solution has exactly one placement covering `c`, so rule (1)+(2) hold. Gain is a factor `|Stab(c)|`: 2 for a corner of a square (the diagonal reflection), `|G|` if `c` is fixed by the whole group (e.g. the centre of an odd square). Hook: `cell_count` is `G`-invariant, so the min-count cells form a `G`-invariant set, and among them one can prefer the cell with the largest stabilizer -- `_cell_priority` is exactly where that preference goes. Orbits here mix blocks, so the orbit code works on gids, not (block, index).

**C. Depth-gated symmetry.** Whichever of A/B is used, only test for symmetry near the root (or when a cheap invariant says it is plausible). See section 5: the per-node sweep cost is real and the payoff is concentrated at the top of empty boards.

Not worth it: orbit-elimination with dedupe (section 3).

## 5. Measured data (old code vs. the cell solver)

Before the cleanup, block-MRV with `symmetry_branching` vs the flat cell-MRV solver (best of runs; empty boards capped at 15 s; "sols/s" for symmetry runs is inflated because derived solutions are cheap transforms):

| case | block pp+pb | block pp+pb+sym | cell (no partition) |
|---|---|---|---|
| `main_puzzles/50` (full) | 0.61 s, 4618 placed | 0.64 s, 4557 placed | 0.26 s, 3114 placed |
| `pyramid_puzzles/100` (full) | 1.62 s, 12890 placed | 1.93 s, 12876 placed | 0.60 s, 7934 placed |
| `empty_main` (15 s) | 37 sols/s | 146 sols/s | 127 sols/s |
| `empty_pyramid` (15 s) | 0 sols | 0 sols | 11.5 sols/s |

Reading it:
- **Preplaced puzzles get essentially nothing from symmetry**: the first preplaced letter breaks the board's symmetry. `main_puzzles/50`: 15 of 101 solutions derived by symmetry; `pyramid_puzzles/100`: 0 of 6, yet the per-node sweep cost about +19% (1.62 s -> 1.93 s) for 14 saved placements.
- **Empty boards are where it pays**: `empty_main` first 300 solutions took 63102 placements without symmetry vs 11202 with, 227 of the 300 derived. That is the case the cell solver only matches (127 vs 146 sols/s) rather than beats, and `empty_pyramid` has never been tried with symmetry at all.
- The symmetry that matters is at the top of the tree. After a few placements the open region is almost never symmetric, which is why depth-gating (C) should recover most of the gain at almost none of the cost. Test this rather than assume it.
- Nested symmetry works (symmetries found at successive depths multiply): the `quadrant_puzzle` test went from 97 placements to 15 with 21 derived; the `open_pocket_puzzle` derives 3 of its 4 solutions.

## 6. Porting the old machinery to flat ids

- **Placement image.** A symmetry is `image` (`n_cells,` permutation of compact cells, identity outside the region). Use `image_ext = np.append(image, n_cells)` so the padding cell maps to itself. For a set of candidate gids, `np.sort(image_ext[_P[gids]], axis=1)` is the image's cell set; because the padding index is the largest, padding stays at the end after sorting. Build once a dict `{sorted padded row bytes: gid}` over all `_P` rows (the old `_placement_lookup`, keyed by gid instead of block-local index) and look the image rows up in it. The lookup is guaranteed to hit for a live placement and a region symmetry.
- **Orbits.** Same as the old `_orbits`: iterate candidates in visit order, the first unvisited one represents its orbit, and the image that takes it to each other member falls out of the same sweep. Visit order now comes from the lexicographic sort, so the representative is the best-ranked member, not the lowest index -- fine, the logic is order-agnostic.
- **Derived solutions.** Old `_apply_symmetry(solution, image)`: `transformed.grid[image] = solution.grid`, and `chosen_placement_idx` remapped through the placement image (block-local index = `gid - _lo[pos]`). Transforming the whole grid is safe because every image is the identity outside the region, so earlier decisions, preplaced blocks and pockets elsewhere are untouched. Unlike the old code, the derived solution must also be a `game.copy()` snapshot with `chosen_placement_idx` consistent (`check_solution_consistent` in `test_symmetry.py` verifies this).
- **Yield structure.** The old loop yielded `solution` then `_apply_symmetry(solution, image)` for each `image` in the representative's orbit, inside the recursion over the representative. `rec` is already a generator, so it drops in unchanged.
- **Region symmetry per node.** `Setup.region_symmetries(available_spots)` with a memo keyed on `available_spots.tobytes()` (the same leftover region recurs across siblings). `available_spots` in `rec` is the compact bool array of open cells.
- **Seeds.** The representative is the first orbit member in visit order, and priorities are per-solve constants, so results stay reproducible for a given seed; the *order* solutions arrive in will differ from an unsymmetric run (derived solutions arrive in bursts right after their representative's).

## 7. Testing and benchmarking hooks

- Correctness: `check_matches_reference` (solver vs. the independent DFS, for seeds `None`/0/7) on the open pocket, quadrants, 30 random tilings, `main_puzzles` 40-60, `pyramid_puzzles` 80-90; `pyramid_puzzles/100` pinned at 6 solutions. Add a derived-solution counter (the old `CountingSolver` counted `_apply_symmetry` calls and placements) to assert the payoff on the quadrant and pocket puzzles, and keep `NodeCountCheckingSolver` (incremental counts vs. recount) running.
- Benchmarking: `plotting.benchmark` compares configs given as option dicts forwarded to the solver (`DEFAULT_CONFIGS = [{}]`). Once `Solver.solve` takes an option such as `symmetry`, benchmark it with `run_benchmark(puzzle, configs=[{}, {"symmetry": True}], nr_tests=..., T=...)` on `empty_main`, `empty_pyramid`, `main_puzzles/50` and `pyramid_puzzles/100`. Old runs saved with the three legacy flags still load (they are folded into `options`).
