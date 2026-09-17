# Solver symmetry modes (3 and 4) — design notes

Branch: `compact-grid`. Mode 3 was built in an earlier session; mode 4 (this
document's main subject) generalises it and closes the gap mode 3 left open.

## The idea

When the still-open region of the board has a symmetry, only search one branch
per symmetric group of candidate placements and derive the rest by transforming
the solutions the searched branch returns, instead of re-searching them.

Both modes implement that same shape. They differ only in *where the symmetries
come from*.

## Mode 3 — whole-board symmetries, filtered

**`Setup.board_symmetries`** — every rigid symmetry of the whole board's cell
shape under its lattice's isometries, as a permutation of compact cell indices.
Pure geometry, computed once, cached, shared like `placements`.

**`Solver._active_symmetries(available_spots)`** — the subset of those that fix
every currently-decided cell *pointwise*, recomputed at each node. Always
includes the identity.

### The limitation this leaves

`_active_symmetries` can only ever filter the precomputed **global** list. It
cannot find a symmetry local to the open region's own shape that isn't the
restriction of some whole-board rigid motion.

Concretely: a 4x2 board with `A D / B C` decided in the left 2x2 and a 2x2 open
pocket on the right. The pocket plainly has a 4-fold symmetry, but no rigid
motion of the whole 4x2 board permutes it while leaving `A,B,C,D` alone, so
mode 3 sees only the identity and gets **zero** benefit. That is the common case
on real puzzles, which is why the original 32-puzzle sweep showed no speedup:
heavily-preplaced puzzles have their board symmetry broken by the first
preplaced letter, before search even starts.

## Mode 4 — the open region's own symmetries

**`Lattice.point_group`** — every integer matrix `M` with
`M.T @ gram2 @ M == gram2`. Pure lattice geometry, no board involved: 8 matrices
for the square lattice, 48 for the pyramid's. Cached on the leaf class.

**`Setup.region_symmetries(mask)`** — every symmetry of an *arbitrary*
sub-region of the board's cells, as compact-index permutations that are the
**identity outside the region**. `board_symmetries` is now just this called on
an all-True mask, so there is one implementation rather than two.

**`Solver`**: `_region_symmetries` (memoized wrapper, capped at
`_SYM_CACHE_MAX`), `_build_placement_lookup` / `_placement_image` (cell-set ->
placement index; replaced the old precomputed `_build_placement_perms`, which
mode 4 can't use and which was also the expensive part of mode 3's setup),
`_orbits` (shared by both modes), `_apply_symmetry` (now takes an image array
rather than an index into the global list).

### Making the per-node check cheap

The first working version spent **76% of mode 4's runtime inside
`region_symmetries`** (~1.7ms a call: a Python loop over the point group doing
matrix multiplies, bounding-box normalisation and `ravel_multi_index` per
matrix). Four changes took it to ~4% on `empty_main` and ~20% on the pyramid:

1. **Precomputed tables instead of coordinate arithmetic**
   (`Setup._symmetry_tables`). Every cell's position under every point-group
   matrix is flattened once into a *padded* grid, sized from the actual
   extremes of `_point_group_coords` so nothing can wrap around an axis.
   Flattening is linear and order-preserving there, so a translation becomes a
   scalar add and "lexicographically smallest cell" becomes `min()`. At a node
   the whole group is one gather of shape `(K, m)`.
2. **Off-board hits map to a sentinel** (`n_cells`, one past the end) rather
   than `-1`, so they can be tested by the same lookup instead of their own
   pass.
3. **No cell-set comparison at all.** The map is injective (invertible matrix,
   injective translation and lookup), so `m` distinct cells landing inside a
   region of `m` cells *is* onto. One membership gather replaces a sort plus
   compare — the single biggest win after the tables.
4. **Skip forced moves.** When the branching block has one candidate, its orbit
   is itself whatever the region's symmetry is, so the sweep is pure cost.
   `min(counts)` picks the most constrained block, so this is common: it
   removed ~30% of sweeps on pyramid/100.

The memo cache absorbs most of what is left (the same leftover hole recurs
across sibling branches). Remaining cost is essentially numpy call overhead on
`(K, m)` arrays, with `K = 8` on the square lattice and `K = 48` on the
pyramid's — which is why the pyramid keeps a visible overhead and `main`
doesn't.

### Why the search is cheap: at most one translation per matrix

The region's symmetry group is a subgroup of point group x translations, which
sounds expensive to search at every node. It isn't. **A finite set admits no
nontrivial translational self-symmetry** — if `x -> Mx + t1` and `x -> Mx + t2`
both map region `R` onto `R`, composing one with the other's inverse gives a
pure translation stabilising a finite set, so `t1 == t2`. So for each `M` there
is at most one candidate `t`, and normalising both shapes to their bounding-box
corner (`S - S.min(axis=0)`, the standard polyomino canonicalisation) recovers
it directly. The whole group falls out of one sweep over `point_group`.

### Why it's a strict generalization

Any whole-board symmetry that fixes the decided cells pointwise necessarily maps
the open region onto itself, so it turns up in the region sweep too. Mode 4's
group always contains mode 3's.

It also subsumes the "value-invariance instead of pointwise-identity"
refinement that was noted as future work: a global symmetry preserving decided
*values* still maps the open region onto itself, so its restriction there is
found anyway.

## Correctness

1. **No duplicates, nothing missing.** Within an orbit, `other != rep` as cell
   sets, so a transformed grid can never equal the one it came from. For a fixed
   `g` with `g(rep) = other`, `S -> g(S)` is a bijection from completions of
   `R \ rep_cells` onto completions of `R \ other_cells`, so the derived
   solutions are exactly the ones the skipped branch would have produced.

2. **Nesting needs no bookkeeping.** Unlike mode 3, a child node's region group
   is *not* a subgroup of its parent's — that is the entire point. It doesn't
   matter: correctness is a plain induction. A node is correct if its recursive
   call is complete and duplicate-free, and the node's own image is the identity
   outside its region, so transforming a whole completed grid can't disturb what
   an ancestor or a sibling component decided. Symmetries found at successive
   depths simply multiply out — total solutions emitted along a path is the
   product of its orbit sizes.

3. **Safe with component splitting.** In the split path `rec` is called with
   `available_spots = comp_mask`, so the sweep naturally computes the symmetry
   of that component alone — exactly the symmetric-pocket case. Components
   filled later by the `on_complete` chain are disjoint from the region.

4. **Every placement image exists.** `Setup._valid_orientations` enumerates
   every matrix satisfying the block's gram condition and
   `_compute_placement_indices` every on-board translation of each, so composing
   a lattice isometry with a valid placement always lands on another one. The
   `_placement_image` lookup is guaranteed to hit.

5. **Images are deduped by their permutation of the region**, not by matrix:
   distinct matrices that act identically on the region's cells act identically
   on placements, so collapsing them is free. This is why e.g. a domino region
   reports 2 symmetries rather than its group's order of 4.

## Bug found and fixed while testing this

`solve(seed=...)` shuffled `self.placements` **in place**. That dict is
`Setup.placements`, shared by reference with every other Puzzle in the book — so
a seeded solve silently invalidated every already-recorded
`chosen_placement_idx` (the preplaced blocks) on its own game *and* on every
other puzzle sharing that Setup. Present in all modes including 0–2, unrelated
to symmetry, but it corrupts exactly what modes 3/4 remap. Fixed in
`Solver._shuffle_placements`: the solver now gets a private shallow copy of the
Setup with its own reordered placements, and re-points its own already-placed
blocks at the rows they moved to. Shuffling an index array rather than the rows
draws the same random stream, so a given seed still yields the same ordering.

## Testing

`python test_symmetry.py` — geometry and group axioms for `region_symmetries`,
the 4x2 open-pocket case, the 4x4 four-quadrant case (nested symmetry), 30
randomly-generated tilings, the `main_puzzles` 40–60 / `pyramid_puzzles` 80–90
ranges and `pyramid_puzzles/100` (the hardest pyramid puzzle, and mode 4's worst
case for per-node cost), all asserting **identical solution sets** across modes, plus
per-solution consistency between `grid` and `chosen_placement_idx` (which is
what `_apply_symmetry` could break without changing a grid). Ends with an
informational payoff table.

Note for future perf work: the 40–60 / 80–90 ranges are good for *correctness*
regression but say nothing about *payoff*, being symmetry-poor by construction.
Use the empty/near-empty puzzles or the synthetic symmetric setups instead.

## Measured payoff

Placements actually tried (i.e. search work) and wall-clock, from
`test_symmetry.py`'s payoff table (interleaved repeats, best of each — mode
order otherwise biases the result badly):

| case | mode 2 | mode 3 | mode 4 |
|---|---|---|---|
| 4x2 open pocket | 8 | 8 | **2** |
| 4x4 quadrants | 97 | 25 | **15** |
| main_puzzles/50 | 7 714 (0.57s) | 7 714 (0.56s) | 7 547 (0.58s) |
| pyramid_puzzles/100 | 23 238 (1.73s) | 23 238 (1.92s) | 23 207 (2.14s) |
| empty_main, first 300 solutions | 107 666 (7.72s) | 45 810 (3.39s) | **18 055 (1.46s)** |

`empty_main` is the headline: **5.3x faster than mode 2 and 2.3x faster than
mode 3**, with 227 of the 300 solutions derived by transform rather than
searched. The open-pocket row is the gap mode 3 could not close at all.

Where mode 4 has nothing to find it now costs **+3% on main_puzzles/50** and
**+24% on pyramid_puzzles/100** (down from +475% before the optimisation round
— that case went 11.5s -> 2.14s). The pyramid keeps a visible overhead because
its lattice point group has 48 matrices against the square lattice's 8, so each
node's sweep is six times as wide.

## Still out of scope

- **Cross-component symmetry** — a symmetry of the open board mapping one
  connected component onto a *different* one (two mirror-image pockets, neither
  symmetric alone). `region_symmetries` would find it, but the split path
  returns before the branching step ever runs, so it goes unused. Exploiting it
  means deduping the block-to-component partitionings under the component
  permutation each symmetry induces, inside the `get_all_partitions` loop.
- **Further performance tuning.** After the round above, the per-node sweep is
  down to numpy call overhead on `(K, m)` arrays and the profile is dominated by
  work mode 4 shares with mode 2 (`scipy.ndimage.label`,
  `_prune_placement_masks_incremental`, `partitioning.find_subsets`). What is
  left that is specific to symmetry: narrowing `K` on the pyramid (most of its
  48 matrices tilt any region off the slope, but knowing which needs the
  region); sharing the memo cache across Solver instances on the Setup, which
  would help repeated benchmark trials; and skipping the sweep at nodes whose
  remaining subtree is too small for an orbit to repay it.
- **Composable arguments** (`prune` / `branch_components` / `symmetry` flags)
  instead of a growing mode-number enum.
