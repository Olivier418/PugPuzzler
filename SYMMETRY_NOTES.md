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

`Solver.solve(branching=..., symmetry=..., up_to_symmetry=...)`.

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

Full enumeration, identical solution sets, best of 3.

| case | `branching="cell"` | default (balanced+symmetry) | speedup |
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

## 6. Where the old code lives

The geometry restored here (`Lattice.point_group`, `Setup.region_symmetries`,
`_symmetry_tables`) came from `git show bf25303:classes/lattice.py` and
`git show bf25303:classes/puzzle.py`. The last committed solver with the
per-node version is `343f69e` (`Solver.py`, "mode 4"), and its design notes are
`git show 343f69e^:SYMMETRY_SOLVER_NOTES.md` -- still the best account of the
per-node sweep's cost work if section 4 is ever revisited. The partitioning of
section 5 is `git show 343f69e:partitioning.py` plus modes 1 (prune) and 2
(branch) in `git show 343f69e:Solver.py`.
