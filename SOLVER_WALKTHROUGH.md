# Reading the solver

A guide to `classes/kernel.py` and `classes/solver.py`, written to be read with
both files open. Every concrete number below was printed from the running code
on **IQpuzzler `main_puzzles/45`**, so you can reproduce any of it.

---

## 0. The shape of it, in one paragraph

A node branches on the scarcest **item** — either an empty cell (try every
placement covering it) or an unplaced block (try every placement of it) —
because either way the branches partition that node's solutions. The board's
occupancy is a **bitmask**: `occ` is one `uint64` per 64 cells, so "does this
placement fit" is `pmask[gid] & occ == 0`, one AND. Nothing is maintained
incrementally; counts are recomputed when a rule asks for them, capped so the
scan stops as soon as a candidate cannot win.

There is no symmetry reduction and no partitioning into components. Both were
built, measured and removed — `SOLVER_NOTES.md` has the numbers and the
reasoning, which is the file to read before adding either back.

---

## 1. The five nouns

Everything in both files is built from these. Get them and the rest follows.

| term | what it is | example on `main_puzzles/45` |
|---|---|---|
| **cell index** | `0 .. n_cells-1`, a compact index into the board's real cells (no off-board padding). Same indexing `State.grid` uses. | `n_cells = 55` |
| **block position** | `0 .. n_blocks-1`. The kernel's own numbering of blocks. **Not** the `Block` key your JSON uses — `solver.block_ids[pos]` translates. | `n_blocks = 12`, `block_ids = [0..11]` |
| **global id (`gid`)** | One specific placement: one block, in one orientation, at one spot. Numbered `0 .. n_total-1`. | `n_total = 1789` |
| **`occ`** | `uint64[NW]`. Bit *c* set = cell *c* is filled. | start: `0x47ffffff`, 28 of 55 cells |
| **`used`** | one `uint64`. Bit *p* set = block position *p* is placed. | start: `0b001100010111`, 6 of 12 |

**The gid ↔ block relationship.** Each block's placements are one contiguous
run. `block_start = [0, 264, 528, 736, ...]`, so block position 0 owns gids
0–263, position 1 owns 264–527, and so on. Therefore:

```
pos            = pblock[gid]              # which block
local index    = gid - block_start[pos]   # which of that block's placements
```

That local index is exactly what `State.place_unchecked(block_key, local_idx)`
wants, which is how a solution gets rebuilt (§6).

**A worked placement.** `gid = 0` is block position 0, covering cells
`[0, 1, 2, 7]`:

```
pmask[0] = 0x0000000000000087
         = 0b...0001 0000111
                 ^      ^^^
              cell 7   cells 2,1,0
```

Read bit 0 on the **right**. That 64-bit integer is the entire representation
of that placement's geometry.

---

## 2. `kernel.Tables` — the static data

Built once per `Setup` by `build_tables()` and cached on it as
`Setup.kernel_tables`, so a whole book builds them once rather than once per
puzzle (worth ~2x on a book solve). ~70 KB. Immutable; never written during a
search.

| field | shape / type | meaning |
|---|---|---|
| `pmask` | `(1789, 1)` uint64 | `pmask[gid]` = that placement's cells as bits. **The core of the whole design.** |
| `pblock` | `(1789,)` uint8 | `pblock[gid]` = its block position |
| `pcells` | `(1789, 5)` int16 | its cell indices as plain numbers, `-1` padded. Only `rank` needs this; the search uses `pmask`. |
| `psize` | `(1789,)` uint8 | how many of those 5 slots are real |
| `cell_start`, `cell_pl` | `(56,)`, `(8259,)` int32 | **CSR list**: the placements covering cell *c* are `cell_pl[cell_start[c] : cell_start[c+1]]`. Cell 0 has 41 of them, starting `[0, 30, 60, 96, ...]`. |
| `block_start` | `(13,)` int64 | the run boundaries above |
| `full` | `(1,)` uint64 | all `n_cells` bits set = `0x007fffffffffffff`. A board equal to this is solved. |
| `neigh` | `(55, 1)` uint64 | `neigh[c]` = c's neighbouring cells as bits. Only `order="pockets"` uses it. |
| `n_cells`, `n_blocks` | int | 55, 12 |
| `fan` | int | widest candidate list, `max(cell degree, block run)` = 264. A block branch's list is a whole block run, so both kinds fit the same buffers. |

"CSR" is just a flat array plus offsets — the standard way to store lists of
different lengths in two arrays instead of a list of arrays (numba can't take a
list of arrays).

---

## 3. `kernel.Workspace` — one search's mutable state

Built by `make_workspace()` **fresh at every kernel entry**, never stored on the
Solver. That is deliberate: it is why running `solve()` twice on the same
`Solver` gives byte-identical output with no reset logic, and why abandoning the
generator half-way (`itertools.islice`) leaks nothing.

The stack is indexed by **depth** = how many blocks we have placed so far.

| field | shape | meaning |
|---|---|---|
| `occ` | `(NW,)` uint64 | cells filled *right now* |
| `used` | `(1,)` uint64 | blocks placed right now (an array, not a scalar, because the kernel mutates it) |
| `ctl` | `(3,)` int64 | `[depth, need_frame, stamp_counter]` — the three loop variables that must survive between calls |
| `st_gid` | `(13,)` int32 | `st_gid[d]` = the placement chosen at depth *d*. The path. |
| `st_cands` | `(13, 264)` int32 | `st_cands[d]` = the candidate list at depth *d* |
| `st_n` | `(13,)` int32 | how many candidates that is |
| `st_cur` | `(13,)` int32 | how many we have already tried |
| `sol_buf`, `sol_len` | `(64, 13)`, `(64,)` int32 | completed paths, waiting to be handed back |
| `cnt`, `stamp` | `(55,)` | per-node memo for `order=True` (§5) |
| `key`, `idx` | `(264, 6)`, `(264,)` | scratch for the sort |

`st_cands[d]`, `st_n[d]`, `st_cur[d]` together are one **stack frame**: "at this
depth, here are the moves, here is how far through them I am."

---

## 4. `kernel.kernel()` — the search loop

This is the part worth reading slowly. It is one `while True` with four
branches, and it is iterative rather than recursive for two reasons: numba's
recursion support segfaulted on it, and an explicit stack can be **paused and
resumed**, which is what lets the Python side stay a lazy generator with an
exact time limit.

The invariant, true at the top of every iteration:

> `occ` and `used` reflect exactly the placements `st_gid[0 : depth]`, and
> `need_frame` says whether depth `depth`'s candidate list has been built yet.

Because that holds at the top of every iteration, it holds at every `return`
too — so resuming is literally "call it again and let the loop continue." There
is no separate resume path that could be wrong.

### Branch 1 — build a frame (`need_frame == 1`)

```python
solved = all(occ[v] == full[v])
```

**Why full board = solved.** `Setup._validate_area` guarantees the blocks' total area
equals the board's, and pre-placed blocks fill whole placements. So covering
every cell with distinct unplaced blocks *must* have used all of them — and the
converse too, which is what makes `branch="block"` terminate: place every block
and `occ == full` follows. No separate "every block placed" check is needed.

If solved: copy `st_gid[0:depth]` into `sol_buf`, then **zero the frame**
(`st_n[depth] = 0`) so this solution cannot be emitted twice on re-entry. Return
`FULL` if the buffer is full.

Otherwise pick an item and list its moves:

```python
kind, i = choose_item(...)               # cell or block to branch on -> §5
if kind == 0: n = collect_cell(t, i, occ, used, nw, st_cands[depth])
else:         n = collect_block(t, i, occ, used, nw, st_cands[depth])
rank(...)                                # sort them                  -> §5
```

**Everything after those three lines is identical either way.** The rest of the
loop only ever sees "a list of gids", never what kind of item produced it —
which is exactly why `order` applies uniformly to both (§5).

On puzzle 45's opening position `choose_item` returns **cell 29**, with 11 live
placements: `[799, 831, 1215, 1255, 1295, 1329, 1383, 1410, 1464, 1518, 1745]`.
Those are the 11 children of the root node.

### Branch 2 — out of budget

```python
if nodes >= budget: return BUDGET
```

The only reason the kernel ever stops early. The driver picks `budget` so this
happens about every 2 ms (§6).

### Branch 3 — backtrack (frame exhausted)

```python
if st_cur[depth] == st_n[depth]:
    depth -= 1
    if depth < 0: return DONE
    occ ^= pmask[st_gid[depth]]     # undo
    used ^= BIT[pblock[st_gid[depth]]]
```

**`^=` undoes the placement.** XOR with the same mask that was OR'd in clears
exactly those bits, because we know they were set. No saved copies, no undo log
— the placement is its own inverse. `depth < 0` means we have backtracked past
the root: the subtree is finished.

### Branch 4 — descend (take the next candidate)

```python
g = st_cands[depth, st_cur[depth]]
st_cur[depth] += 1
occ |= pmask[g];  used |= BIT[pblock[g]]
st_gid[depth] = g
nodes += 1;  depth += 1;  need_frame = 1
```

Four instructions of actual work. Everything else in the file exists to decide
*which* `g`.

---

## 5. Choosing and ordering

Two separate decisions, and it is worth keeping them apart:

- **`branch`** — *which item* this node branches on. Changes the shape of the
  tree, so it changes how many nodes exist.
- **`order`** — *what sequence* that item's placements are tried in. Cannot
  change which nodes exist in a full enumeration; it only changes which
  solutions come out first.

### `choose_item(t, occ, used, nw, branch)`

Two MRV scans, each capped at the best count so far, returning `(kind, index)`.

| `branch` | rule |
|---|---|
| `"cell"` | the empty cell with the fewest live placements |
| `"block"` | the unplaced block with the fewest |
| `"both"` **(default)** | whichever of the two is scarcer; cells scanned first, so a cell wins a tie |

**Why the two counts are comparable at 1:1.** A child of either kind places
exactly one block, so an equal candidate count means an equal branching factor
*and* equal progress. There is no unit mismatch for a weight to correct, which
is why `"both"` is a plain minimum with no knob. Timings and node counts
for each mode are in `SOLVER_NOTES.md` §1.

`"block"` alone is the weak one and exists as a baseline: it cannot see an
uncoverable cell until some block runs out of room, so it prunes almost nothing
where little is pre-filled. Worst book puzzle measured is `main_puzzles/65` at
2 s against 0.01 s; on the *empty* main board it finds no solution at all in
30 s.

**The trick that makes MRV affordable** is the `cap` argument to `_count_cell` /
`_count_block`. Only the *minimum* matters, so once an item is known to have at
least as many candidates as the best item so far, the rest of its list is
irrelevant and the count gives up. Cells have ~150 placements each while the
running best is usually in the single digits, so the scan almost always stops
after a handful of tests. This alone took MRV from 243k to ~400–500k nodes/s.

Note that a 0-candidate item is a **dead end** (a cell nothing can cover, or a
block with nowhere to go) and a 1-candidate item is **forced** — both end the
scan immediately, so MRV prunes while it chooses.

### `_live()` — the one predicate

```python
used & BIT[pblock[gid]] == 0   and   pmask[gid] & occ == 0
```

"Its block isn't placed yet, and it doesn't overlap anything." Every candidate
collector calls this. **Do not write a second copy of it** — see §8.

### `rank(...)`

Sorts a candidate list in place, best first. One implementation serves both
branch kinds, and it is never told which it was handed: every key below is a
property of a **placement**, not of the item the node branched on. That is the
whole mechanism behind "`order` works the same for cells and blocks" — there is
nothing to keep in sync, because there is only one code path.

| `order=` | key |
|---|---|
| `True` / `"counts"` | each candidate's cells' live-placement counts, sorted and compared like a tuple |
| `"pockets"` | how many empty cells it would strand with no empty neighbour |
| `"fanout"` | how many options it leaves for the next cell |
| `False` | table order — or, if seeded, by the random priority alone |
| `None` | on iff a limit is set |

Every key ends with `priority[gid]`, a permutation, so no two candidates ever
compare equal. That totality is what makes a seeded run reproducible: without
it, ties would fall out in whatever order the sort happened to leave them.

`rank` returns immediately when `n < 2`. Most nodes have 0 or 1 candidates, so
ordering is nearly free.

`cnt`/`stamp` memoise the per-cell counts *within one node*: `stamp[c] ==
stamp_id` means `cnt[c]` is current. `stamp_id` is `ctl[2]`, which only ever
increments — it must be unique across the **whole run**, not per call, because
`nodes` restarts at zero every chunk.

---

## 6. `solver.Solver` — the Python side

### `solve()` top to bottom

`Solver` is thin — it owns the objects the kernel cannot see, and the generator
contract. Nothing else.

1. **Resolve `branch` and `order`** into integer rule codes (`_resolve_rules`).
2. **Draw priorities.** A local `priority`, one per placement: `np.arange`
   unseeded, `rng.permutation` seeded.
3. **Build the start position** (`_start`). Walk `state.grid` for filled cells
   → `occ0`; mark every already-placed block in `used0`, which is how a
   Puzzle's pre-filled letters are excluded from the search.
4. **Allocate one `Workspace`** and run the chunking loop below.

### The chunking loop

```python
while True:
    if past deadline: return
    status, n_sol, nodes = kernel.kernel(t, w, cap, budget, rank_rule, priority, branch_rule)
    for i in range(n_sol):
        found += 1
        yield self._emit(w.sol_buf[i, :w.sol_len[i]].tolist())
        if found >= max_solutions: return
    if status == DONE: return
```

**The kernel is never told `max_solutions`.** It gets a buffer cap and a node
budget; all counting stays in Python. That is what keeps the generator lazy: a
consumer that stops after three solutions costs three solutions' worth of work,
not a whole enumeration.

`_next_budget` keeps an EMA of nodes/second and asks for ~2 ms of work. That
bounds how far `time_limit` can overshoot (~2 ms) and costs ~0.25% in dispatch.
The EMA persists across `solve()` calls deliberately: a chunk boundary changes
*where the kernel returns*, never the order it visits nodes, so it cannot affect
output.

### `_emit(path)` — gids back into a `State`

```python
for gid in path:
    pos = pblock[gid]
    solution.place_unchecked(block_ids[pos], gid - block_start[pos])
```

`place_unchecked` writes the grid **and** `chosen_placement_idx` together, so a
yielded solution can never have one without the other. That is precisely what
`tests/_helpers.assert_valid_solution` cross-checks.

---

## 7. What is deliberately not here

Two reductions were built, measured and removed. `SOLVER_NOTES.md` has the
numbers; the short version:

**Symmetry (orbit reduction).** At a node whose open region had a nontrivial
symmetry group, branch on a *block*, group its placements into orbits, search
one representative each and transform its solutions into the rest. It had to run
in Python on a "symmetric spine" above the kernel, because the orbit argument
needs a branch set the group carries onto itself — a block's candidate set
qualifies, a cell's does not — plus `Setup.placement_lookup` and State
transforms, none of which exist down here. It was exact and it worked. It also
did **nothing** on real puzzles: across all 221 book puzzles in both shipped
games it changed the node count by at most 1.4%, and on PRO `main_puzzles` not
at all. It paid only on the fully-empty boards, and branching on both item kinds
(§5) since recovered most of that. Removed 2026-09-23.

**Partitioning into components.** Prune or split when the open region falls
apart. Measured at 0.3–0.4% of nodes on most books (16–22% on the PRO boards,
which barely broke even on time), and it arrives too late to matter: MRV already
goes straight for a leftover pocket, which is most of what a split would find.

---

## 8. The sharp edges

Four things that are load-bearing and non-obvious.

1. **`_live` must not be duplicated.** `choose_item` picks the item with the
   fewest live candidates; `collect_cell`/`collect_block` then materialise them.
   If the counters and the collectors ever disagreed, the counts steering MRV
   would be lies — and a count of 0 is taken as *proof* that the node is dead,
   so the search would prune live subtrees with no error. Everything goes
   through `kernel._live` (`_count_block` drops its block half only because its
   caller has already established the block is unplaced).

2. **The `BIT` table instead of `1 << c`.** Under `NUMBA_DISABLE_JIT=1` the
   shift is evaluated by numpy, which (numpy 2.x) promotes `uint64 << int` to
   float64 and destroys the high bits. Indexing a table behaves identically in
   both modes. Same reason every literal is written `np.uint64(...)`.

3. **Shape discipline on `used`.** It is a scalar everywhere except inside the
   Workspace. Passing a `(1,)` array where a scalar was expected once produced a
   `(1,1)` array that numba compiled **34x slower**, with correct results and no
   warning.

4. **No closures around njit functions.** A closure cell gets hashed into
   numba's `cache=True` key, and `benchmark.py`'s trial subprocesses would
   recompile instead of loading the cache. `warmup()` exists so they don't race
   to write it.

---

## 9. Poking at it

**Debug it as plain Python.** This is the single most useful thing here:

```bash
NUMBA_DISABLE_JIT=1 python -m unittest tests.test_branching.TestPentominoes
```

Every `@njit` becomes an ordinary function. `pdb` works, exceptions have real
tracebacks, `print` works, and array bounds are actually checked (numba's are
not). ~1300x slower, so use small cases — but the node counts and solution sets
are identical, which is what makes one implementation enough.

**Watch the search.** Node counts come back from the kernel:

```python
from classes import kernel as K
real = K.kernel; total = [0]
def spy(*a):
    r = real(*a); total[0] += r[2]; return r
K.kernel = spy
```

**Compare branch rules** on any puzzle:

```python
for b in ("cell", "block", "both"):
    print(b, sum(1 for _ in puzzle.solve(branch=b)))
```

**The regression check that matters.** Capture `sorted(s.grid.tobytes() for s
in Solver(p).solve(**mode))` for every book puzzle in both games under every
`branch` x `order` combination, and diff against the same capture from the
previous solver. 221 puzzles x 14 modes is a couple of minutes and is how every
change to the kernel has been verified.

---

## 10. Where to start reading

If you read in this order it should build up cleanly:

1. `kernel.Tables` + `build_tables` — the data, nothing clever
2. `kernel._live`, `collect_cell`, `_count_cell` — three short functions
3. `kernel.kernel` — the loop, with §4 beside it
4. `kernel.choose_item` + `rank` — the two decisions, with §5 beside it
5. `solver.solve` + `_start` + `_emit` — the Python wrapper

Skip §5 on a first pass. The solver is correct and complete without any of it —
`choose_item` and `rank` only decide speed and ordering, never the answer.
