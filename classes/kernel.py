"""The exact-cover search itself: static tables, and an iterative
bitmask DFS compiled by numba.

A leaf module -- it imports numpy and numba and nothing else, so it knows
nothing about Setup/State/Puzzle. classes.solver owns the translation
between this module's flat integer world and those objects.

**Representation.** A cell is a bit. The board's occupancy is `occ`, an
array of `NW = ceil(n_cells / 64)` uint64 words, and a placement is the
same thing precomputed (`pmask[gid]`). "Does this placement fit" is then
one AND per word: the conflict test reads 8 bytes.

That makes the conflict test nearly free, so what a node costs is
`choose_item`, which has to look at the candidate lists to stay
well-informed.

**Branching.** A node branches on the scarcest *item*, and an item is
either an empty cell (try every placement covering it) or an unplaced
block (try every placement of it). Both partition the node's solutions,
so either is sound on its own; taking the minimum over both is strictly
better informed than either, and is the default. See `choose_item`.

`nw` is read from `occ.shape[0]` at runtime and every mask operation
loops over it, so there is no cell-count ceiling anywhere and no separate
code path for big boards: 55 cells (IQpuzzler) and 64 (IQquub, exactly at
the one-word boundary) both give NW == 1, and a 200-cell board would give
NW == 4 and work unchanged. The word loop costs ~1.19x against a
hand-written scalar kernel.

**Global ids.** A placement's global id is its row in `pmask`,
and block position p owns the contiguous run
`block_start[p]:block_start[p + 1]`, so a block-local placement index is
`gid - block_start[p]`. Nothing ever reorders these rows.

**Debugging.** Run under `NUMBA_DISABLE_JIT=1` and every function here
executes as plain Python, with pdb and real exceptions, at roughly 1/1300
the speed. That is the only reason this module can be the single
implementation. It constrains the source: see `BIT` below for the place
where numpy's scalar semantics and numba's would otherwise diverge.
"""
from typing import NamedTuple

import numpy as np
from numba import njit

# Why a table rather than `np.uint64(1) << c`: under NUMBA_DISABLE_JIT the
# expression is evaluated by numpy, which (numpy 2.x) promotes
# `uint64 << python_int` to float64 and silently destroys the high bits,
# while numba would keep it an integer shift. Indexing a precomputed table
# behaves identically in both. Every uint64 literal below is written
# np.uint64(...) for the same reason.
BIT = np.uint64(1) << np.arange(64, dtype=np.uint64)

_U0 = np.uint64(0)

# kernel() return codes.
DONE = 0    # the subtree is exhausted; do not call again
FULL = 1    # sol_buf filled; drain it and call again to continue
BUDGET = 2  # node budget spent; call again to continue

# Which kind of item a node branches on, see choose_item().
BRANCH_CELL = 0   # the empty cell with fewest live placements
BRANCH_BLOCK = 1  # the unplaced block with fewest live placements
BRANCH_BOTH = 2   # whichever of the two is scarcer (the default)

# Ranking rules, see rank().
RANK_NONE = 0     # leave candidates in table order
RANK_PRIORITY = 1  # by the per-solve random priority alone (seeded, order=False)
RANK_COUNTS = 2   # scarcest cells' live counts first (order=True)
RANK_POCKETS = 3  # fewest dead empty cells created
RANK_FANOUT = 4   # most options left for the next cell

_INF = np.int32(np.iinfo(np.int32).max)


class Tables(NamedTuple):
    """Everything derived from a Setup alone. Built once per Setup and
    shared by every Solver on it -- a whole book is one board and one
    block collection, and none of this depends on which cells a
    particular puzzle starts with.

    A NamedTuple because numba accepts one as an argument (verified with
    cache=True) while keeping the call sites readable; it is immutable,
    which is what makes sharing it safe.
    """
    pmask: np.ndarray       # (n_total, NW) uint64  -- placement occupancy
    pblock: np.ndarray      # (n_total,)    uint8   -- block *position*
    pcells: np.ndarray      # (n_total, kmax) int16 -- cell indices, -1 padded
    psize: np.ndarray       # (n_total,)    uint8   -- cells per placement
    cell_start: np.ndarray  # (n_cells + 1,) int32  -- CSR offsets
    cell_pl: np.ndarray     # (incidences,) int32   -- gids through each cell
    block_start: np.ndarray  # (n_blocks + 1,) int64
    full: np.ndarray        # (NW,) uint64          -- all n_cells bits set
    neigh: np.ndarray       # (n_cells, NW) uint64  -- adjacency, for RANK_POCKETS
    n_cells: int
    n_blocks: int
    fan: int                # widest candidate list: max(cell degree, block run)


def build_tables(placement_cells, n_cells, cell_coords, unit_vectors):
    """Tables from a Setup's `placement_cells` (block idx -> (N, k) array
    of compact cell indices), its cell count, and -- for RANK_POCKETS'
    adjacency -- the board position of every compact cell and the
    lattice's unit vectors.

    Returns `(tables, block_ids)`. `block_ids[p]` is the caller's block
    key for block position p -- this module only ever speaks positions.
    """
    block_ids = list(placement_cells)
    n_blocks = len(block_ids)
    if n_blocks > 64:
        # `used` is a single uint64; 12 blocks today, 10 on IQquub.
        raise ValueError(f"{n_blocks} blocks; the used-block mask holds 64.")

    sizes = [placement_cells[b].shape[0] for b in block_ids]
    block_start = np.cumsum([0] + sizes, dtype=np.int64)
    n_total = int(block_start[-1])
    kmax = max(arr.shape[1] for arr in placement_cells.values())
    nw = max(1, -(-n_cells // 64))

    pcells = np.full((n_total, kmax), -1, dtype=np.int16)
    psize = np.zeros(n_total, dtype=np.uint8)
    pblock = np.zeros(n_total, dtype=np.uint8)
    for pos, block_idx in enumerate(block_ids):
        arr = placement_cells[block_idx]
        lo, hi = int(block_start[pos]), int(block_start[pos + 1])
        pcells[lo:hi, :arr.shape[1]] = arr
        psize[lo:hi] = arr.shape[1]
        pblock[lo:hi] = pos

    pmask = np.zeros((n_total, nw), dtype=np.uint64)
    for gid in range(n_total):
        for c in pcells[gid, :psize[gid]]:
            pmask[gid, int(c) >> 6] |= BIT[int(c) & 63]

    # CSR: the placements through each cell, ascending within a cell.
    flat = pcells.ravel()
    real = flat >= 0
    cell_of = flat[real].astype(np.int64)
    gids = np.repeat(np.arange(n_total), kmax)[real]
    order = np.argsort(cell_of, kind="stable")
    cell_pl = gids[order].astype(np.int32)
    cell_start = np.zeros(n_cells + 1, dtype=np.int32)
    cell_start[1:] = np.cumsum(np.bincount(cell_of, minlength=n_cells))

    full = np.zeros(nw, dtype=np.uint64)
    for c in range(n_cells):
        full[c >> 6] |= BIT[c & 63]

    # Two cells are neighbours when their coordinates differ by a lattice
    # unit vector. Used only by RANK_POCKETS.
    neigh = np.zeros((n_cells, nw), dtype=np.uint64)
    lookup = {tuple(row): i for i, row in enumerate(cell_coords)}
    for i, row in enumerate(cell_coords):
        for v in unit_vectors:
            j = lookup.get(tuple(row + v))
            if j is not None:
                neigh[i, j >> 6] |= BIT[j & 63]

    fan = int(max(np.diff(cell_start).max(initial=1), np.diff(block_start).max(initial=1)))
    return Tables(pmask, pblock, pcells, psize, cell_start, cell_pl, block_start,
                  full, neigh, n_cells, n_blocks, fan), block_ids


class Workspace(NamedTuple):
    """The mutable state of one kernel run.

    Allocated fresh at every kernel entry and never stored on the Solver.
    That is deliberate and load-bearing: it is what makes a second
    `solve()` on the same Solver produce bit-identical output without any
    reset discipline, and what makes abandoning the generator part-way
    (itertools.islice) leak nothing.
    """
    occ: np.ndarray       # (NW,) uint64   -- cells filled at the current node
    used: np.ndarray      # (1,)  uint64   -- blocks placed at the current node
    ctl: np.ndarray       # (3,)  int64    -- [depth, need_frame, stamp_counter]
    st_gid: np.ndarray    # (D,)  int32    -- the placement chosen at each depth
    st_cands: np.ndarray  # (D, fan) int32 -- that depth's candidate list
    st_n: np.ndarray      # (D,)  int32    -- its length
    st_cur: np.ndarray    # (D,)  int32    -- how far through it we are
    sol_buf: np.ndarray   # (cap, D) int32 -- completed paths
    sol_len: np.ndarray   # (cap,) int32
    cnt: np.ndarray       # (n_cells,) int32 -- per-node memo of live counts
    stamp: np.ndarray     # (n_cells,) int64 -- which node cnt[c] is valid for
    key: np.ndarray       # (fan, kmax + 1) int32 -- sort keys
    idx: np.ndarray       # (fan,) int32   -- the permutation being sorted


def make_workspace(t, occ0, used0, cap=64):
    """A Workspace positioned at the partial solution `(occ0, used0)`."""
    depth = t.n_blocks + 1
    kmax = t.pcells.shape[1]
    return Workspace(
        occ=np.asarray(occ0, dtype=np.uint64).reshape(-1).copy(),
        # reshape(1), not np.array([used0]): `used0` may arrive as a
        # 0-d scalar or a (1,) array, and wrapping the latter in a list
        # would build a (1, 1) array. numba compiles that 34x slower.
        used=np.asarray(used0, dtype=np.uint64).reshape(1).copy(),
        ctl=np.array([0, 1, 0], dtype=np.int64),  # depth 0, frame not built, stamp 0
        st_gid=np.zeros(depth, dtype=np.int32),
        st_cands=np.zeros((depth, t.fan), dtype=np.int32),
        st_n=np.zeros(depth, dtype=np.int32),
        st_cur=np.zeros(depth, dtype=np.int32),
        sol_buf=np.zeros((cap, depth), dtype=np.int32),
        sol_len=np.zeros(cap, dtype=np.int32),
        cnt=np.zeros(t.n_cells, dtype=np.int32),
        stamp=np.full(t.n_cells, -1, dtype=np.int64),
        key=np.zeros((t.fan, kmax + 1), dtype=np.int32),
        idx=np.zeros(t.fan, dtype=np.int32),
    )


# ---- primitives ---------------------------------------------------------
# Module-level, never closures: a closure cell would be hashed into
# numba's cache key, and benchmark.py spawns a subprocess per trial that
# must hit that cache rather than recompile.

@njit(cache=True)
def _fits(pmask, gid, occ, nw):
    for w in range(nw):
        if pmask[gid, w] & occ[w] != _U0:
            return False
    return True


@njit(cache=True)
def _live(pmask, pblock, gid, occ, used, nw):
    """The one liveness predicate, shared by every candidate collector:
    the placement overlaps nothing already down, and its block is not
    already placed.

    Writing it twice is the sharpest hazard here. `choose_item` picks the
    item with the fewest live candidates and `collect_cell`/
    `collect_block` then materialise them; if the counters and the
    collectors ever disagreed, the counts steering MRV would be lies --
    and a count of 0 is taken as proof that the node is dead. So
    `_count_cell` and `_count_block` must stay exactly this predicate
    (`_count_block` drops the block half only because its caller has
    already established the block is unplaced).
    """
    if used & BIT[pblock[gid]] != _U0:
        return False
    return _fits(pmask, gid, occ, nw)


@njit(cache=True)
def collect_cell(t, c, occ, used, nw, out):
    """Live placements covering cell `c`, in table order. Returns how many."""
    n = 0
    for i in range(t.cell_start[c], t.cell_start[c + 1]):
        gid = t.cell_pl[i]
        if _live(t.pmask, t.pblock, gid, occ, used, nw):
            out[n] = gid
            n += 1
    return n


@njit(cache=True)
def _count_cell(t, c, occ, used, nw, cap):
    """How many live placements cover cell `c`, giving up once `cap` of
    them are known. Same predicate as collect_cell, without materialising
    the list.

    The cap is what makes the MRV scan affordable. Only the *minimum*
    matters, so once a cell is known to have at least as many candidates
    as the best cell so far it cannot win and the rest of its list is
    dead weight -- and cells have ~150 placements each while the running
    best is usually in the single digits, so the scan almost always stops
    after a handful of tests instead of walking the whole list.
    """
    n = 0
    for i in range(t.cell_start[c], t.cell_start[c + 1]):
        if _live(t.pmask, t.pblock, t.cell_pl[i], occ, used, nw):
            n += 1
            if n >= cap:
                return n
    return n


@njit(cache=True)
def collect_block(t, pos, occ, used, nw, out):
    """Live placements of block position `pos`, in table order. The
    block-branch counterpart of collect_cell -- same `_live`, so the two
    candidate sets are filtered identically. Returns how many."""
    n = 0
    for gid in range(t.block_start[pos], t.block_start[pos + 1]):
        if _live(t.pmask, t.pblock, gid, occ, used, nw):
            out[n] = gid
            n += 1
    return n


@njit(cache=True)
def _count_block(t, pos, occ, nw, cap):
    """How many live placements block position `pos` still has, giving up
    once `cap` of them are known -- _count_cell's counterpart, and capped
    for the same reason.

    No `used` argument: the caller only ever asks about an unplaced
    block, so `_live` reduces to `_fits`.
    """
    n = 0
    for gid in range(t.block_start[pos], t.block_start[pos + 1]):
        if _fits(t.pmask, gid, occ, nw):
            n += 1
            if n >= cap:
                return n
    return n


@njit(cache=True)
def choose_item(t, occ, used, nw, branch):
    """The item this node branches on, as `(kind, index)`: kind 0 is an
    empty cell, kind 1 an unplaced block position.

    Both kinds are sound branch sets on their own -- every solution
    covers a given cell exactly once, and places a given block exactly
    once -- so either partitions the node's solutions. They are also
    directly comparable: a child of either kind places exactly one
    block, so an equal candidate count means an equal branching factor.
    That is what lets BRANCH_BOTH simply take the minimum over the two,
    which is strictly better informed than either alone (BRANCH_CELL and
    BRANCH_BLOCK exist as baselines).

    Minimum remaining values, in both loops: the scan is capped at the
    best count so far (see _count_cell), and it prunes for free. A count
    of 0 means the node is dead -- a cell nothing can cover, or a block
    with nowhere left to go -- and returning it yields an empty candidate
    list, which makes the caller backtrack at once. A count of 1 is
    forced, and nothing can be scarcer, so the scan stops there too.

    Cells are scanned first and the comparison is strict, so a cell wins
    a tie. Never returns -1: `kernel` tests `occ == full` before calling
    this, so something is always left.
    """
    best_kind = np.int64(0)
    best = np.int64(-1)
    best_n = 1 << 30

    if branch != BRANCH_BLOCK:
        for c in range(t.n_cells):
            if occ[c >> 6] & BIT[c & 63] != _U0:
                continue
            n = _count_cell(t, c, occ, used, nw, best_n)
            if n == 0:
                return np.int64(0), np.int64(c)
            if n < best_n:
                best_kind = np.int64(0)
                best = np.int64(c)
                best_n = n
                if n == 1:
                    return best_kind, best

    if branch != BRANCH_CELL:
        for pos in range(t.n_blocks):
            if used & BIT[pos] != _U0:
                continue
            n = _count_block(t, pos, occ, nw, best_n)
            if n == 0:
                return np.int64(1), np.int64(pos)
            if n < best_n:
                best_kind = np.int64(1)
                best = np.int64(pos)
                best_n = n
                if n == 1:
                    return best_kind, best

    return best_kind, best


@njit(cache=True)
def _cell_count(t, c, occ, used, nw, cnt, stamp, stamp_id):
    """Live placements through cell `c`, memoised for this node.

    Bounds RANK_COUNTS at one pass per *distinct* cell the candidates
    touch, rather than one per (candidate, cell) pair.
    """
    if stamp[c] == stamp_id:
        return cnt[c]
    n = _count_cell(t, c, occ, used, nw, _INF)
    cnt[c] = n
    stamp[c] = stamp_id
    return n


@njit(cache=True)
def _pocket_score(t, gid, occ, nw):
    """Empty cells that placing `gid` would strand: cells left empty with
    every neighbour filled. Nothing can ever cover one, so a candidate
    that makes them is a dead end one ply early."""
    dead = 0
    for c in range(t.n_cells):
        w, b = c >> 6, c & 63
        if occ[w] & BIT[b] != _U0 or t.pmask[gid, w] & BIT[b] != _U0:
            continue  # cell is filled, or about to be
        blocked = True
        for v in range(nw):
            if t.neigh[c, v] & ~(occ[v] | t.pmask[gid, v]) != _U0:
                blocked = False
                break
        if blocked:
            dead += 1
    return dead


@njit(cache=True)
def _fanout_score(t, gid, occ, used, nw):
    """Live placements left for the first cell still empty after `gid`
    goes down. Fewer means the branch is closer to forced."""
    used = used | BIT[t.pblock[gid]]  # gid's own block is placed too
    n_after = 0
    for c in range(t.n_cells):
        w, b = c >> 6, c & 63
        if occ[w] & BIT[b] != _U0 or t.pmask[gid, w] & BIT[b] != _U0:
            continue
        for i in range(t.cell_start[c], t.cell_start[c + 1]):
            g2 = t.cell_pl[i]
            if used & BIT[t.pblock[g2]] != _U0:
                continue
            ok = True
            for v in range(nw):
                if t.pmask[g2, v] & (occ[v] | t.pmask[gid, v]) != _U0:
                    ok = False
                    break
            if ok:
                n_after += 1
        return n_after
    return n_after


@njit(cache=True)
def rank(t, cands, n, occ, used, rule, priority, cnt, stamp, stamp_id, key, idx):
    """Sort `cands[:n]` in place, best first.

    One implementation for both branch kinds, and it never learns which
    it was handed: every rule below keys off a *placement*, not off the
    item the node branched on. That is what makes the `order` option
    uniform across cell and block branches for free.

    Every rule ends in `priority[gid]`, which is a permutation, so the
    order is total and no two candidates ever compare equal. That matters
    for reproducibility: a non-total order would let the representative
    depend on call history.

    RANK_PRIORITY has no branch of its own below on purpose -- it is the
    fall-through, leaving every key column at _INF so only that final
    priority column decides. It is what a seeded `order=False` solve uses.
    """
    if n < 2 or rule == RANK_NONE:
        # Survivors per node are mean 1.0, median 0 -- this short-circuits
        # the large majority of nodes, which is why ranking is cheap.
        return
    nw = occ.shape[0]
    kmax = t.pcells.shape[1]
    width = kmax + 1

    for i in range(n):
        gid = cands[i]
        idx[i] = i
        for j in range(width):
            key[i, j] = _INF
        if rule == RANK_COUNTS:
            # Each cell's live-placement count, sorted ascending and
            # compared like a tuple, so a placement covering the scarcest
            # cells sorts first.
            m = 0
            for j in range(t.psize[gid]):
                c = t.pcells[gid, j]
                if c >= 0:
                    key[i, m] = _cell_count(t, c, occ, used, nw, cnt, stamp, stamp_id)
                    m += 1
            for a in range(1, m):          # insertion sort, m <= kmax <= 8
                v = key[i, a]
                b = a - 1
                while b >= 0 and key[i, b] > v:
                    key[i, b + 1] = key[i, b]
                    b -= 1
                key[i, b + 1] = v
        elif rule == RANK_POCKETS:
            key[i, 0] = _pocket_score(t, gid, occ, nw)
        elif rule == RANK_FANOUT:
            key[i, 0] = _fanout_score(t, gid, occ, used, nw)
        key[i, width - 1] = priority[gid]

    # Insertion sort over the permutation, comparing key rows. n is the
    # live fan-out (p95 of 5, max 28 at a cell; a few hundred at a block),
    # so this beats anything with setup cost. No packing into a single
    # integer: IQquub has kmax 8 and cell degrees over 300, which needs
    # 72 bits.
    for a in range(1, n):
        ia = idx[a]
        b = a - 1
        while b >= 0:
            ib = idx[b]
            cmp = 0
            for j in range(width):
                if key[ib, j] != key[ia, j]:
                    cmp = 1 if key[ib, j] > key[ia, j] else -1
                    break
            if cmp <= 0:
                break
            idx[b + 1] = ib
            b -= 1
        idx[b + 1] = ia

    # Apply the permutation. It needs a buffer -- writing cands[i] =
    # cands[idx[i]] in place would clobber entries still to be read -- and
    # key's last column is free now that the sort is done, so it serves as
    # one rather than allocating (numba would heap-allocate per node).
    for i in range(n):
        key[i, width - 1] = cands[idx[i]]
    for i in range(n):
        cands[i] = key[i, width - 1]


# ---- the search ---------------------------------------------------------

@njit(cache=True)
def kernel(t, w, cap, budget, rule, priority, branch):
    """Depth-first search of the subtree at `w`'s current position.

    Returns `(status, n_sol, nodes)`. `status` is DONE (subtree
    exhausted), FULL (`sol_buf` holds `cap` solutions) or BUDGET (`nodes`
    reached `budget`). For FULL and BUDGET the caller drains `sol_buf` and
    calls again to continue; `w` carries the whole stack, so re-entry is
    literally "run the loop again" and there is no separate resume path
    to get wrong.

    Explicit stack rather than recursion: numba's recursion support
    segfaulted on this search, and a stack is what makes the run
    resumable, which is what lets classes.solver stay a lazy generator
    with an exact time limit.

    Loop invariant, true at the top of every iteration and therefore at
    every return: `occ`/`used` reflect exactly `st_gid[0:depth]`, and
    `need_frame` says whether depth `depth`'s candidate list is built.
    """
    nw = w.occ.shape[0]
    depth = w.ctl[0]
    need_frame = w.ctl[1]
    n_sol = 0
    nodes = 0

    while True:
        if need_frame == 1:
            # Is the board full? That alone means solved: Setup._validate_area
            # guarantees the blocks' total area equals the board's, and
            # pre-placed blocks fill whole placements, so a cover of every
            # cell by distinct unplaced blocks has necessarily used all of
            # them. The converse holds too, which is what makes
            # BRANCH_BLOCK terminate: place every block and occ == full.
            solved = True
            for v in range(nw):
                if w.occ[v] != t.full[v]:
                    solved = False
                    break
            if solved:
                for j in range(depth):
                    w.sol_buf[n_sol, j] = w.st_gid[j]
                w.sol_len[n_sol] = depth
                n_sol += 1
                # Empty the frame before returning, so the solution cannot
                # be emitted twice if we come back here. occ still carries
                # the last placement; the pop branch below unapplies it.
                w.st_n[depth] = 0
                w.st_cur[depth] = 0
                need_frame = 0
                if n_sol == cap:
                    w.ctl[0] = depth
                    w.ctl[1] = need_frame
                    return FULL, n_sol, nodes
                continue

            # Branch on the scarcest item, cell or block. Everything
            # below this point is identical either way -- which is what
            # makes `rank` (and so the `order` option) apply uniformly to
            # both kinds of candidate list.
            kind, i = choose_item(t, w.occ, w.used[0], nw, branch)
            if kind == 0:
                n = collect_cell(t, i, w.occ, w.used[0], nw, w.st_cands[depth])
            else:
                n = collect_block(t, i, w.occ, w.used[0], nw, w.st_cands[depth])
            # The memo stamp has to be unique per node across the WHOLE
            # run, not per call: `nodes` restarts at 0 every chunk, so
            # deriving the stamp from it would let a node reuse a stale
            # cnt[c] left by a same-numbered node of an earlier chunk.
            # ctl[2] is int64 and only ever increments.
            w.ctl[2] += 1
            rank(t, w.st_cands[depth], n, w.occ, w.used[0], rule, priority,
                 w.cnt, w.stamp, w.ctl[2], w.key, w.idx)
            w.st_n[depth] = n
            w.st_cur[depth] = 0
            need_frame = 0
            continue

        if nodes >= budget:
            w.ctl[0] = depth
            w.ctl[1] = need_frame
            return BUDGET, n_sol, nodes

        if w.st_cur[depth] == w.st_n[depth]:
            depth -= 1
            if depth < 0:
                w.ctl[0] = depth
                w.ctl[1] = need_frame
                return DONE, n_sol, nodes
            g = w.st_gid[depth]
            for v in range(nw):            # a placement is its own undo
                w.occ[v] ^= t.pmask[g, v]
            w.used[0] ^= BIT[t.pblock[g]]
            continue

        g = w.st_cands[depth, w.st_cur[depth]]
        w.st_cur[depth] += 1
        for v in range(nw):
            w.occ[v] |= t.pmask[g, v]
        w.used[0] |= BIT[t.pblock[g]]
        w.st_gid[depth] = g
        nodes += 1
        depth += 1
        need_frame = 1


_warm = False


def warmup(t):
    """Compile/load every njit function, on a throwaway one-node budget.
    Idempotent: only the first call in a process does anything.

    The first kernel() call in a process pays numba's cache load (~0.35 s)
    or, if this file changed, a full compile (seconds). Callers that time
    a solve run this before starting the clock, so that cost isn't billed
    to the first puzzle. benchmark.py also calls it in the parent: its
    per-trial subprocesses would otherwise race to write
    classes/__pycache__/kernel-*.nbi and each pay the compile.

    Numba specializes on argument types, not sizes, so any Tables will do.
    """
    global _warm
    if _warm:
        return
    priority = np.arange(t.pmask.shape[0], dtype=np.int32)
    w = make_workspace(t, np.zeros(t.full.size, dtype=np.uint64), np.uint64(0), cap=1)
    kernel(t, w, 1, 1, RANK_COUNTS, priority, BRANCH_BOTH)
    _warm = True
