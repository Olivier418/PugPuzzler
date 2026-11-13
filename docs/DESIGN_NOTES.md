# Design notes: why the solver is the way it is

`SOLVER_WALKTHROUGH.md` explains how the solver works. This file covers the
decisions behind it: what was measured, and the ideas that were built, tested
and left out, so that anyone considering them again can see how they fared.

**Two solvers.** The project's first solver was written in NumPy and kept, for
every cell and block, a running count of the placements still available,
updating those counts on every move. It was replaced by the current one
(`pugpuzzler/classes/kernel.py`), a numba-compiled depth-first search on
bitmasks that recounts at every node instead and is about 2.9x faster over a
full enumeration of both games. A few results below were measured on the first
solver and are marked *(first solver)*. Their absolute times can't be compared
with the current solver's, but they are kept because the conclusions drawn
from them still hold.

Absolute timings on the device that was used drifted by up to 3x between runs, so every table
below reports **node counts** alongside time, and the node counts are the
figures to trust. Timings are best-of-3, interleaved.

---

## 1. Branching: cells, blocks, or both

A node branches on the scarcest **item**. An item is either an empty cell (try
every placement covering it) or an unplaced block (try every placement of it).
Either partitions the node's solutions: every solution covers a given cell
exactly once, and places a given block exactly once, so either is sound alone.

The two counts are directly comparable at 1:1, with no weight to tune: a child
of either kind places exactly one block, so an equal candidate count means an
equal branching factor *and* equal progress. (Keying on each item's count
*relative* to its count at the root was measured on the first solver and was
worse everywhere, up to 5.5x: MRV's whole justification is that the candidate
count **is** the branching factor, and a ratio does not bound it.)

Full enumeration of each book, and a fixed solution count on the empty boards:

| case | both, time / nodes | cell, time / nodes |
|---|---|---|
| IQpuzzler `main_puzzles` (3680 sols) | **0.55 s / 204833** | 0.62 s / 260916 |
| IQpuzzler `pyramid_puzzles` (41) | **0.04 s / 10568** | 0.11 s / 36916 |
| PRO `main_puzzles` (40) | **0.07 s / 21015** | 0.07 s / 21767 |
| PRO `pyramid_puzzles` (44) | **0.69 s / 233264** | 0.75 s / 264095 |
| `empty_main`, 2000 sols | **0.49 s / 204692** | 1.67 s / 685857 |
| `empty_pyramid`, 500 sols | **1.59 s / 548864** | 14.35 s / 4853760 |

`both` wins or ties everywhere, so it is the default. The worst case for it is
`main_puzzles/62` at +4% nodes against `cell`, which is inside the noise;
`pyramid_puzzles/100` is the other extreme, 2685 nodes against 16325.

`block` alone is the weak one, and exists only as a baseline. It cannot see an
uncoverable cell until some block runs out of room, so with little pre-filled
there is almost nothing to prune: worst book puzzle measured is
`main_puzzles/65` at 2 s against 0.01 s, and on the *empty* main board it finds
no solution at all in 30 s. `tests/test_solver.py` keeps it off open boards
for that reason.

**What makes MRV affordable** is the `cap` argument to `_count_cell` /
`_count_block`. Only the minimum matters, so a count gives up as soon as the
item cannot beat the best so far. Cells have ~150 placements each while the
running best is usually single digits, so the scan almost always stops after a
handful of tests. This alone took MRV from 243k to ~400–500k nodes/s.

**Incremental counts are not worth it.** The alternative to recounting is
maintaining every item's count as blocks go down, as the first solver did.
Measured on IQpuzzler `main_puzzles` 40–60: the capped scan costs ~156 `_fits`
tests per node (one AND each, all in L1), while each placement kills ~44 live
placements, i.e. ~490 counter updates per node counting the undo, plus the work
of finding those kills (conflict lists, or a placements² bitset that outgrows
the cache). The cap already made recounting cheaper than bookkeeping.

## 2. `order`: ranking a node's candidates

Sorting each node's candidates cannot change a full enumeration's node set at
all: sibling order does not decide which nodes exist. It does find the first
solutions markedly sooner (time-to-first summed over both IQpuzzler books:
0.37 s with, 0.60 s without, *(first solver)*). It also costs ~25% of runtime.

So it is an option, defaulting to on exactly when `time_limit` or
`max_solutions` is set, which is precisely when finding solutions early is
worth paying for.

One implementation (`kernel.rank`) serves both branch kinds, and it is never
told which it was handed: the key is a property of a *placement*, not of the
item the node branched on. That is why `order` behaves identically under all
three `branch` modes, with nothing to keep in sync.

**Removed: two more ranking rules.** `"pockets"` ranked a candidate
by how many empty cells it would strand with no empty neighbour (rejecting a
dead branch one ply early; it needed a per-cell adjacency table, `Tables.neigh`),
and `"fanout"` by how many options it left for the next empty cell. Benchmarked
on the five empty boards, 15 trials of ~15 s each per rule (`benchmark.py`,
throughput in solutions/s pooled over trials, and median time to the first
solution):

| board | counts (`True`) | `False` | pockets | fanout |
|---|---|---|---|---|
| IQpuzzler `empty_main` | 3370 / 4.9 ms | **4487** / 9.2 ms | 4190 / 9.8 ms | 4118 / 10.5 ms |
| IQpuzzler `empty_pyramid` | 374 / 91 ms | **490** / 12 ms | 478 / 12 ms | 444 / 66 ms |
| PRO `empty_alt` | **4196** / 15 ms | 4073 / 10 ms | 3894 / 11 ms | 3734 / 8.4 ms |
| PRO `empty_main` | 3062 / 14 ms | 4460 / 8.8 ms | 4299 / 9.3 ms | **5062** / 5.9 ms |
| PRO `empty_pyramid` | 252 / 22 ms | 248 / 18 ms | 224 / 20 ms | **336** / 345 ms |

No rule wins on every board, and plain table order (`False`) is best or close
on most, so the two extra rules weren't worth their ~60 lines of kernel. Note
that `counts` (what `order=None` picks when a limit is set) is also the
slowest on three of the five by throughput. It is the default under a limit
because it tends to find the first solutions sooner, not because it finds the
most per second: for long sampling runs, `order=False` can be the better
choice.

---

## 3. Removed: the symmetry (orbit) reduction

**What it did.** At a node whose open region had a nontrivial symmetry group,
branch on a *block*, group its placements into orbits under the group, search
one representative per orbit, and produce the other members' solutions by
transforming the representative's. `up_to_symmetry` yielded just the
representatives: one solution per symmetry class.

**It was correct.** Every solution places the branch block exactly once, so the
branches partition the solutions, and a symmetry of the open region is a
bijection from the completions of one branch onto those of its image. It was
verified exact against published pentomino counts (12 pentominoes on 3x20: 8
solutions, 2 up to the rectangle's symmetry) and on self-symmetric fixtures.

**Pros.**

- Large on fully-empty boards, the only place a board-sized symmetry group
  survives the first few placements.
- `up_to_symmetry` was a genuine feature, not just a speedup, and it was exact
  and free: strictly less work than enumerating and de-duplicating.
- Cost nothing where the open region was asymmetric, beyond one group lookup
  per solve.

**Cons, and why it went.**

- **It did nothing on real puzzles.** Across all 221 book puzzles in both
  shipped games it moved the node count by at most 1.4%, and on PRO
  `main_puzzles` not at all: identical node counts, i.e. the reduction never
  fired once.
- It forced a Python-side "symmetric spine" above the kernel, with its own
  block-branching, orbit and transform machinery, because the orbit argument
  needs a branch set the group carries onto itself (a block's candidates
  qualify; a cell's do not) plus a lookup from a transformed placement back to
  its index and transforms of the board state, none of which can exist inside
  the kernel.
- ~500 lines across the solver, the setup and a separate lattice-geometry
  module, plus a 286-line test file, to maintain for that.

Measured before removal, full enumeration per book:

| case | sym=True, time / nodes | sym=False, time / nodes |
|---|---|---|
| IQpuzzler `main_puzzles` | 0.556 s / 257140 | 0.545 s / 260916 |
| IQpuzzler `pyramid_puzzles` | 0.100 s / 36906 | 0.100 s / 36916 |
| PRO `main_puzzles` | 0.065 s / 21767 | 0.063 s / 21767 |
| PRO `pyramid_puzzles` | 0.737 s / 264089 | 0.727 s / 264095 |
| `empty_main`, 2000 sols | 0.201 s / 71680 | 1.539 s / 685857 |
| `empty_pyramid`, 500 sols | 0.451 s / 90910 | 13.357 s / 4853760 |

**What the removal actually cost.** Branching on both item kinds (§1) landed at
the same time and recovered most of the empty-board loss, which was the only
place symmetry paid. Net, today's solver against the symmetric one:

| case | before (cell + symmetry) | after (both, no symmetry) | |
|---|---|---|---|
| IQpuzzler `main_puzzles` | 0.556 s | 0.546 s | 1.02x faster |
| IQpuzzler `pyramid_puzzles` | 0.100 s | 0.036 s | 2.8x faster |
| PRO `main_puzzles` | 0.065 s | 0.068 s | tie |
| PRO `pyramid_puzzles` | 0.737 s | 0.694 s | 1.06x faster |
| `empty_main`, 2000 sols | 0.201 s | 0.494 s | 2.5x slower |
| `empty_pyramid`, 500 sols | 0.451 s | 1.590 s | 3.5x slower |

Every real puzzle ties or gets faster; the two fully-empty standalone boards pay
2.5x and 3.5x. `up_to_symmetry` is gone with it.

**If it is ever wanted back**, the cheap version is a post-filter:
canonicalise each finished solution under the board's symmetry group and keep
the first of each class. That needs only the geometry (the board's symmetry
group and a search for the symmetries of the open region), not the orbit machinery, but it does
strictly more work than enumerating, and it cannot be made exact under
`max_solutions`, which is why it was not kept.

## 4. Removed: partitioning into components *(first solver)*

Not a symmetry, but the same kind of question: what to do when the open region
falls apart into connected components. Two variants were measured: **prune**
the node when the unplaced block sizes cannot be distributed over the component
sizes, and **split**, which tiles the smallest component once, groups its
tilings by the block set used, and solves the rest once per set.

| case | nodes saved, prune | prune time | nodes saved, split |
|---|---|---|---|
| `main_puzzles` | 0.35% | 1.13x | -0.1% |
| `pyramid_puzzles` | 0.3% | 1.25x | -0.05% |
| PRO `main_puzzles` | 16% | 1.00x | 11% |
| PRO `alt_puzzles` | 22% | **0.86x** | 17% |
| PRO `pyramid_puzzles` | 0.4% | 1.20x | 0.0% |
| empty boards (sampled subtrees) | 0.2–1.0% | slower | -0.3 to +0.2% |

Three reasons it does not transfer to an MRV search:

- MRV already goes straight for a leftover pocket, whose cells are the scarcest
  on the board. Most splits cut off a 3–5 cell pocket that one block fills,
  which is exactly what the branch does anyway.
- Splits come late, mostly 7–9 placements deep, where subtrees are small.
  Fewer than 100 of ~12,000 sampled splits happen within 4 placements, so
  checking only near the root finds almost nothing.
- The check costs 11–28 µs against 70–100 µs of node work, so even PRO
  `main_puzzles`' 16% fewer nodes only breaks even. PRO's diagonal `alt` board,
  which splits constantly, is the one place it wins.
