# Symmetry in the solver -- what was built, and what was left out

Supersedes the pre-implementation notes (the earlier version of this file, and
`git show 343f69e^:SYMMETRY_SOLVER_NOTES.md`), which were written against the
old block-MRV solver and whose numbers do not transfer to the current one.

## 1. The rule

Skipping a branch and deriving its solutions from another branch is safe
exactly when the branch set `C` satisfies both:

1. `C` is carried onto itself by every symmetry being used;
2. every solution contains **exactly one** member of `C`.

Placements of **one block**: both hold for the whole group. Placements covering
**one cell**: (2) holds, (1) only for symmetries that fix that cell -- which for
a corner of an 11x5 board is only the identity. That is why cell-MRV cannot use
symmetry, and why the solver branches on a *block* at exactly the nodes where a
group is live.

## 2. What the solver does

`Solver.solve(symmetry=..., up_to_symmetry=..., order=...)`. An ordinary node
branches on the scarcest item (section 6); what matters here is only that a
symmetric node branches on a **block** instead.

- `Setup.region_symmetries` is called **once per solve**, on the root's open
  cells. Not per node.
- Branch on a block, group its live placements into orbits, search one
  representative per orbit, derive the rest with `_transform`.
- The child's group is the representative's **stabilizer**, which
  `Solver._orbits` returns from the same sweep. No geometry is recomputed.
  Trivial stabilizer (the common case) means that child and its whole subtree
  revert to ordinary branching, permanently.
- The group only ever shrinks down a branch, and it dies per-branch, not
  globally -- different root branches lose it at different depths.

Images are permutations of compact cell indices that are the identity outside
the root's open region. A stabilizer element maps the representative's cells
onto themselves, and those cells all carry the same block, so transforming a
whole solution grid leaves them -- and everything outside the open region,
including preplaced blocks -- untouched. Nothing has to be re-restricted at
depth.

### Why `up_to_symmetry` is exact

Solutions split by the branching block's placement; orbits of placements pair
up with orbits of solutions; within a representative's branch the induction
repeats under the stabilizer. So the searched solutions are exactly one per
symmetry class, with no dedupe pass. It is also strictly *less* work than
`symmetry=True`, since no image is ever built.

## 3. Measured

Full enumeration, identical solution sets, best of 3. Measured against the
solver of `7427453`, i.e. before the item branch and the flat-table rewrite of
section 6; `"balanced"` was renamed `"hybrid"` in that commit.

`branching` was a solver argument then. The modes it selected were removed on
2026-09-23 (see the end of section 6), so this table and the mode comparisons
in section 6 are a record of how the decision was reached, not something the
current solver can reproduce.

| case | `branching="cell"` | then-default (hybrid+symmetry) | speedup |
|---|---|---|---|
| 12 pentominoes, 3x20 (8 solutions) | 3.13 s | 0.99 s | **3.15x** |
| 12 pentominoes, 4x15 (1472 solutions) | 66.2 s | 9.8 s | **6.73x** |
| `main_puzzles` whole book (3680 solutions) | 12.14 s | 11.89 s | 1.02x |
| `pyramid_puzzles` whole book (41 solutions) | 1.51 s | 1.53 s | 0.99x |

Board groups: main `|G| = 4`, pyramid `|G| = 8`.

The books get nothing, as expected: 20 of 72 `main_puzzles` and 6 of 29
`pyramid_puzzles` do have `|G| > 1`, but they are the heavily-preplaced easy
ones (3-15 open cells) and account for 3.1% and 1.4% of each book's solve time.
What the books *do* get is `up_to_symmetry`: every symmetric `main_puzzle`
collapses to a single solution (11 -> 1 of 8, 12 -> 1 of 4, 17-20 -> 1 of 4,
35/65 -> 1 of 2; only 22 has 2), which is the uniqueness the game claims.

Choosing the branching block: `min(block_count)` — fewest live placements — is
right, and it was worth checking. Picking the block with the best orbit
reduction instead searches strictly fewer solutions and is still **2.5x
slower**, because a block with few placements constrains the board far more and
that dominates:

| case | rule | root block | reduction | time |
|---|---|---|---|---|
| 3x20 | fewest placements | X (18 placements) | 2.00x | **0.79 s** |
| 3x20 | best orbit reduction | V (72) | 4.00x | 1.96 s |
| 4x15 | fewest placements | X (26) | 3.71x | **12.4 s** |
| 4x15 | best orbit reduction | V (104) | 4.00x | 28.5 s |

Fewest-placements gives up some reduction at the root (main block L: 2.70x of a
possible 4x), but that is exactly what the stabilizer chain earns back -- a
placement with an undersized orbit is one that *preserves* part of the
symmetry, so the region left open is still symmetric and the reduction nests.

## 4. Deliberately not done: symmetric pockets

Symmetries of the open region that are **not** inherited from the root group --
a symmetric hole left in an otherwise asymmetric board. That is the old
"mode 4". Finding them needs `region_symmetries` at every node, and in the
current lean solver that does not pay:

- Sweep cost: **34 us/node** (main) / **52 us/node** (pyramid) against only
  **77 / 99 us/node** of actual node work, i.e. **+54-57%** runtime
  unmemoized, ~+9% with a memo at a 67% hit rate.
- Headroom, measured as the share of the tree under an outermost symmetric node
  weighted by `1 - 1/|G|`, *after* the root group is already exploited:
  **19.4%** best case on `empty_main` (1234 symmetric nodes of 43799), 14.6% on
  `main_puzzles/50`. And that bound ignores the cost of block-branching at
  those nodes.
- Only 5.8-7.2% of nodes are symmetric, mostly `|G| = 2`, concentrated at small
  leftover regions whose subtrees are tiny.

On both empty boards the outermost symmetric node **is the root**, so the whole
tree already sits under the reduction the solver does perform.

The old numbers (+3% on `main_puzzles/50`, +24% on `pyramid_puzzles/100`) looked
affordable only because the old block-MRV solver did ~5x more work per node.

Also checked and ruled out: **congruent blocks**. Two identically-shaped pieces
would give a free "swap them" symmetry with no geometry involved, but there are
none in either IQpuzzler setup. (IQpuzzlerPRO and IQquub were not checked.)

## 5. Also measured and not done: partitioning into components

Not a symmetry, but the same kind of question. The old block-MRV solver had two
options for when the open region falls apart into connected components:
**prune** the node if the unplaced block sizes can't be distributed over the
component sizes, and **branch** on every such distribution, solving each
component with only its own blocks. Re-measured on 2026-09-21 with counting-only
copies of the search built on `Solver`'s own tables, `_place` and branch
candidates (identical solution counts in every mode):

- *prune*: sizes only, as before, or with each block restricted to the
  components it has a live placement in (prunes marginally more).
- *split*: stronger than the old branch. Tile the smallest component once with
  any blocks, group its tilings by the block set used, solve the rest once per
  set and multiply. The old version re-solved the later component once per
  solution of the earlier one, so this is an upper bound on it.

| case | nodes saved, prune | prune time | nodes saved, split |
|---|---|---|---|
| `main_puzzles` | 0.35% | 1.13x | -0.1% |
| `pyramid_puzzles` | 0.3% | 1.25x | -0.05% |
| PRO `main_puzzles` | 16% | 1.00x | 11% |
| PRO `alt_puzzles` | 22% | **0.86x** | 17% |
| PRO `pyramid_puzzles` | 0.4% | 1.20x | 0.0% |
| all five empty boards (sampled subtrees) | 0.2-1.0% | slower | -0.3 to +0.2% |

Times are best-of-N for the cheapest check tried: the open region as a Python
int, flood-filled by masked shifts, then a subset sum over the unplaced block
sizes. A BFS over neighbour lists costs about the same.

It did help the block solver (block branching still gets ~20% fewer nodes from
it, and is still ~1.6x slower than cell branching). It doesn't transfer:

- Cell-MRV already goes straight for a leftover pocket, whose cells are the
  scarcest. Most splits cut off a 3-5 cell pocket that one block fills, which is
  exactly what the cell branch does anyway.
- Splits come late: mostly 7-9 placements deep on the empty boards, where
  subtrees are small. Fewer than 100 of ~12,000 sampled splits happen within 4
  placements, so checking only near the root finds almost nothing.
- The check costs 11-28 us against 70-100 us of node work, so even PRO main's
  16% fewer nodes only breaks even. PRO alt's diagonal board, which splits
  constantly, is the one place it wins.

Where decomposition clearly works is an even split near the root, and only
PRO `empty_alt` produces those at its symmetric (block-branched) nodes. On
`[19, 31]`, *split* takes 70k nodes down to 13k, and it is 2-5.5x on five of the
ten such nodes, but together they are an estimated 0.4% of that tree. On
IQpuzzler `empty_main` the symmetric nodes never split.

## 6. The item branch and the flat-table rewrite (2026-09-22)

Not symmetry, but it changes every number above. Solution sets are identical to
`7427453`'s: both IQpuzzler books, all three IQpuzzlerPRO books and IQquub,
under `cell`/`hybrid`/`item` x `symmetry`, seeded, and `up_to_symmetry`.

**Items.** "Every cell covered exactly once" and "every block placed exactly
once" are the same constraint, so cells and blocks now share one index space
and one `counts` array (`_Tables.placement_items` appends the block's own
column to each placement's cells). One decrement and one argmin then do the
dead-end check, the solved test and the branch choice at once, for both kinds.
`branching="item"` (then the new default, now the only rule) branches on
whichever item is scarcest -- a block the moment fewer placements are left for
it than for any cell, which plain cell-MRV cannot see. `"hybrid"` was kept as
the baseline it had become, until the measurements at the end of this section
retired it.

**Forward checking.** `_place` picks the child's branching item itself and
rejects the child if any item is left with no placement. Those children used to
be entered and only then found dead: 72k of `main_puzzles`' 190k search calls,
each having first paid for a recursion, a grid write and a full node's
bookkeeping. Covered items hold `_COVERED` rather than 0, which is what lets
one argmin distinguish "scarcest", "dead" and "solved" -- and it retired
`open_cells` and `_dead_end` entirely.

**Cheaper nodes.** `_Tables` depends on the Setup alone, so a book builds it
once instead of once per puzzle (`_tables_for`, weakly keyed). `kill(gid)`
merges the overlap set with the placement's own block, so `_place` is two fancy
indexes, two array ops and a `bincount`; `live` is copied only once the child
survives. The search carries a `path` of global ids and materialises a State
only at a solution, so no grid is touched per node.

Best of 3, interleaved old/new in one process (this laptop's timings drift up
to 3x otherwise). "nodes" counts the children the search descends into.

| case | old cell | old default | new cell | new default | speedup | nodes |
|---|---|---|---|---|---|---|
| `main_puzzles` (3680 solutions) | 10.62 s | 10.56 s | 2.31 s | **1.88 s** | **5.6x** | 157k -> 71k |
| `pyramid_puzzles` (41) | 1.33 s | 1.33 s | 0.30 s | **0.10 s** | **14.0x** | 20k -> 4k |
| PRO `pyramid_puzzles` | 10.19 s | 10.28 s | 2.31 s | **2.14 s** | **4.8x** | 171k -> 71k |
| 12 pentominoes, 3x20 (8) | 2.80 s | 0.87 s | 0.61 s | **0.21 s** | **4.1x** | 17k -> 5k |

On `main_puzzles` the two effects split roughly 80/20: forward checking removes
the 72k dead children, and the item branch takes the nodes that actually branch
from 81k to 67k (17%). Symmetry still pays on top of both (new cell vs new
default: 1.2x on `main_puzzles`, 3.0x on `pyramid_puzzles`).

### Empty boards, and where symmetry actually pays

Seconds to a fixed number of solutions (best of 3), which is the honest
stand-in for a rate: a time limit would measure whatever region the search
happened to be in.

| case | old default | new default | new, `symmetry=False` | new `cell` |
|---|---|---|---|---|
| `empty_main`, 2000 solutions | 1.79 s | **0.57 s** | 1.79 s | 4.85 s |
| `empty_pyramid`, 500 solutions | 6.83 s | **2.32 s** (1.73 s with `order=False`) | 13.50 s | 36.86 s |

So the empty boards gain less from the rewrite (2.9-3.1x) than the books
(4.8-14x) -- the books are full of tight preplaced puzzles, which is where
forward checking and a scarce *block* pay most -- but they are the one place
the symmetry framework itself is worth anything: **3.1x** on main and **5.8x**
on the pyramid (`|G|` = 4 and 8), against **1.00-1.02x** on the books, exactly
as section 3 predicted. It costs nothing where it can't be used.

Sections 4 and 5 (symmetric pockets, partitioning) got *less* attractive, not
more: a node now costs ~27 us on `main_puzzles` where it cost ~77 us, while the
per-node geometry sweep (34-52 us) and the partition check (11-28 us) are
unchanged. Both were already below break-even.

### Cell vs block: what "most constrained" means

Branch on `min(cell counts, w * block counts)`; `w = 1` is what ships,
`w = inf` is cells-only. The *relative* variant instead keys on each item's
count divided by its count at the root. Nodes are children the search descends
into; every row carries the same float-key overhead, so the times compare with
each other but not with the shipped solver.

| case | | 0.25 | 0.5 | 0.75 | **1** | 1.5 | 2 | cells only | relative |
|---|---|---|---|---|---|---|---|---|---|
| `main_puzzles` | nodes | 1.14 | 0.99 | **0.98** | 1.00 | 1.10 | 1.18 | 1.20 | 1.30 |
| | time | 5.22 | 3.48 | 3.01 | **2.94** | 3.24 | 3.71 | 3.54 | 5.56 |
| `pyramid_puzzles` | nodes | 1.54 | 0.93 | **0.90** | 1.00 | 1.35 | 1.90 | 2.56 | 1.89 |
| | time | 0.29 | 0.14 | **0.13** | 0.15 | 0.21 | 0.30 | 0.44 | 0.40 |
| PRO `main_puzzles` | nodes | 1.92 | **0.94** | 0.97 | 1.00 | 1.01 | 1.01 | 1.01 | 5.53 |
| | time | 0.87 | 0.31 | **0.30** | **0.30** | 0.31 | 0.31 | 0.33 | 2.50 |
| `empty_main` (2000 sols) | nodes | **0.70** | 0.80 | 0.87 | 1.00 | 1.03 | 1.03 | 1.03 | 1.08 |
| | time | 0.73 | **0.69** | 0.70 | 0.75 | 0.93 | 0.86 | 0.78 | 1.04 |

**Absolute counts, compared at 1:1.** The two branch sets are directly
comparable because every child of either kind places exactly one block, so an
equal count means an equal branching factor *and* equal progress; there is no
unit mismatch a threshold would have to correct. Relative is worse everywhere
(1.3x on `main_puzzles`, 5.5x on PRO `main_puzzles`): MRV's whole justification
is that the candidate count **is** the node's branching factor, and a ratio
doesn't bound it -- it will happily pick an item with 20 children over one with
3 because 20 is a smaller fraction of where it started.

A mild block preference (`w` = 0.5-0.75) is a few percent better on three of
the four cases and a few percent worse on `main_puzzles`; not worth a knob.
Note also that node counts flatter block-heavy settings: a block branch tries
many more placements that `_place` rejects, and a rejection costs without
counting as a node -- which is why `w = 0.25` wins on nodes for `empty_main`
and still loses on time.

**`order`.** Sorting each node's candidates (`_order_gids`) is ~25% of the
runtime and cannot change a full enumeration's node set at all -- sibling order
doesn't decide which nodes exist. It does find the first solutions markedly
sooner (time-to-first summed over both IQpuzzler books: 0.37 s with, 0.60 s
without). So it became an option, defaulting to on exactly when `time_limit` or
`max_solutions` is set.

### Retiring the other modes (2026-09-23)

The three alternatives were kept as benchmark baselines until they were
measured head to head against `item`, which section 6 above never did -- it
only established that the solution sets agree. Best of 3, interleaved in one
process, full enumeration on the books and a fixed solution count on the empty
boards. "nodes" is again the children the search descends into.

| case | item | hybrid | cell | block | hybrid/item time | hybrid/item nodes |
|---|---|---|---|---|---|---|
| `main_puzzles` (3680 sols) | **1.99 s** | 2.42 s | 2.43 s | -- | 1.22x | 1.20x |
| `pyramid_puzzles` (41) | **0.097 s** | 0.300 s | 0.293 s | 0.587 s | **3.08x** | 2.56x |
| PRO `main_puzzles` (40) | **0.211 s** | 0.214 s | 0.208 s | 1.722 s | 1.01x | 1.01x |
| PRO `pyramid_puzzles` (44) | **2.14 s** | 2.31 s | 2.34 s | -- | 1.08x | 1.07x |
| `empty_main`, 2000 sols | **0.535 s** | 0.561 s | 4.73 s | -- | 1.05x | 1.03x |
| `empty_pyramid`, 500 sols | **1.93 s** | 2.06 s | -- | -- | 1.07x | 1.06x |

`item` wins every case on both time and nodes, so `branching` was dropped and
the argument with it. Four things the run showed:

- **`hybrid` is `cell`** on `pyramid_puzzles`, PRO `main_puzzles` and PRO
  `pyramid_puzzles` -- the same node counts to the unit (10117, 6258, 76265).
  Its block branch only fires while a group is live, which on those books is
  never at a node that costs anything. It separates from `cell` only on
  `main_puzzles` (84766 vs 85877) and `empty_main` (13252 vs 123884).
- The **closest call** is PRO `main_puzzles`, where `cell` comes in 1.4% under
  `item` (0.208 s vs 0.211 s). That is inside this laptop's noise, and `item`
  still visits fewer nodes there (6178 vs 6258). A tie, not a loss.
- **`block` alone** is 8.2x slower than `item` on PRO `main_puzzles` (1.722 s,
  23068 nodes against 6178) and 6.0x on `pyramid_puzzles`, confirming section
  5's "markedly weaker".
- `empty_main`'s `cell` column (4.73 s against 0.535 s) is **symmetry's**
  payoff rather than the branch rule's -- `cell` is the one mode that could
  never use the orbit reduction. That is why `symmetry` stayed a knob while
  `branching` went: it is the only remaining way to measure the reduction, and
  the tests' only baseline that derives no solution from another.

## 7. Where the old code lives

The four branching modes last coexist in `c398a8b` -- `git show
c398a8b:classes/solver.py` is the solver every number in section 6 was measured
on, including the table above. `cell`/`block`/`balanced` on the pre-item solver
are in `git show 7427453:classes/solver.py`, which is what section 3 measured.

The geometry restored here (`Lattice.point_group`, `Setup.region_symmetries`,
`_symmetry_tables`) came from `git show bf25303:classes/lattice.py` and
`git show bf25303:classes/puzzle.py`. The last committed solver with the
per-node version is `343f69e` (`Solver.py`, "mode 4"), and its design notes are
`git show 343f69e^:SYMMETRY_SOLVER_NOTES.md` -- still the best account of the
per-node sweep's cost work if section 4 is ever revisited. The partitioning of
section 5 is `git show 343f69e:partitioning.py` plus modes 1 (prune) and 2
(branch) in `git show 343f69e:Solver.py`.
