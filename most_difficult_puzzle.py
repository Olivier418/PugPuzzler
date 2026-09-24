"""Search for puzzles that are strictly more difficult than a given
`start` puzzle: fewer solutions and/or more empty spaces, never worse on
either.

Since Setup._validate_area guarantees the blocks' total area equals the board's,
nr_empty_spaces of a candidate depends only on *which* blocks are
pre-placed, never on *which placement* is used -- so the search is really
just the 2**n_blocks subsets of blocks (same space regardless of how many
placements each block has), pruned first by that free empty-spaces count
(a subset with fewer empty spaces than `start` can never beat it, so it's
skipped before any solving happens).

For each surviving subset, this is thorough rather than sampled: every way
to place that subset's blocks without overlapping each other is tried (see
_enumerate_placement_combos), each realized as a candidate and solved for
its true nr_solutions, keeping the best (lowest) found. Three things keep
that tractable:
  - subsets are visited in decreasing nr_empty_spaces order, so every
    frontier member found so far already has at least as many empty
    spaces as whatever's being searched next -- which means its
    nr_solutions alone is a valid ceiling, tighter than `start`'s once
    anything better than `start` has actually been found (see
    most_difficult_puzzles);
  - each realization's solve is capped at that ceiling (tightening further
    to the best found so far within the same subset -- see
    _best_realization), so a losing realization is abandoned the moment
    that's established rather than counted out exhaustively;
  - overlap conflicts reject most placement combinations before they ever
    reach the solver (_enumerate_placement_combos).

A candidate only survives into the result if it actually beats `start`
(at least as good on both axes, strictly better on one); what's returned
is only the mutually non-dominated subset of those survivors -- the
Pareto frontier, as a (PuzzleBook, SolutionBook, SolveStatsBook) triple --
each tagged with `difficulty` (an argument to most_difficult_puzzles, default
constants.FRONTIER_DIFFICULTY) for plotting.plot_puzzlebook, and named A, B,
C, ... in order of decreasing nr_empty_spaces.
"""
import itertools
from typing import NamedTuple

import numpy as np
from tqdm import tqdm

from classes import Puzzle, PuzzleBook, SolutionBook, SolveStatsBook, State
from constants import FRONTIER_DIFFICULTY
from solving import solve_puzzlebook


def _enumerate_placement_combos(setup, block_idxs: tuple[int, ...]):
    """Yield every way to place all of `block_idxs` on `setup`'s board
    without overlapping each other, as {block_idx: placement_idx} dicts.
    Says nothing about whether the rest of the board can still be tiled by
    the remaining blocks -- that's the caller's job (see
    most_difficult_puzzles).

    Backtracks over blocks ordered by ascending placement count (fewest
    choices first), which prunes overlap conflicts as early as possible --
    this, not a clever bound, is what keeps the search tractable."""
    ordered = sorted(block_idxs, key=lambda idx: len(setup.placement_cells[idx]))
    n = len(ordered)
    chosen = {}
    occupied = np.zeros(setup.n_cells, dtype=bool)

    def backtrack(i):
        if i == n:
            yield dict(chosen)
            return
        block_idx = ordered[i]
        for placement_idx, cells in enumerate(setup.placement_cells[block_idx]):
            if occupied[cells].any():
                continue
            occupied[cells] = True
            chosen[block_idx] = placement_idx
            yield from backtrack(i + 1)
            occupied[cells] = False
            del chosen[block_idx]

    yield from backtrack(0)


def _best_realization(setup, subset: tuple[int, ...], ceiling: int):
    """The lowest nr_solutions achievable by any non-overlapping placement
    of `subset`'s blocks, and the placements that achieve it -- or None if
    no realization gets to `ceiling` solutions or fewer at all. `ceiling`
    is the caller's choice of "nothing above this is worth finding" (see
    most_difficult_puzzles: it's the running frontier's own tightest
    nr_solutions, not just `start`'s, which is what lets the search
    tighten itself as it goes).

    Each realization's solve is capped at the best count still worth
    finding (`ceiling` to start, tightening to the current best after
    that), so a realization that's already worse is abandoned the moment
    that's established rather than counted out exhaustively -- this is
    what makes 'the true minimum' tractable at all.

    Stops early the moment the true minimum, 1, is found."""
    best = None  # (nr_solutions, placements)
    cap = ceiling + 1

    for placements in _enumerate_placement_combos(setup, subset):
        state = State(setup)
        for idx, placement_idx in placements.items():
            state.place_unchecked(idx, placement_idx)
        count = sum(1 for _ in state.solve(max_solutions=cap))
        if count >= cap:
            continue  # capped -- not an improvement, exact value doesn't matter
        if count == 0:
            continue  # valid packing, but leaves the rest untileable
        best = (count, placements)
        cap = count  # only a strict improvement on this is worth finding now
        if count == 1:
            break
    return best


class Candidate(NamedTuple):
    """A puzzle that survived the search, with its difficulty axes carried
    alongside it rather than recomputed from the grid or stored on Puzzle."""
    puzzle: Puzzle
    nr_empty: int
    nr_solutions: int


def _dominates(a: Candidate, b: Candidate) -> bool:
    """True if `a` is at least as difficult as `b` (>= empty spaces, <=
    solutions) and strictly more difficult on at least one."""
    at_least_as_hard = a.nr_empty >= b.nr_empty and a.nr_solutions <= b.nr_solutions
    strictly_harder = a.nr_empty > b.nr_empty or a.nr_solutions < b.nr_solutions
    return at_least_as_hard and strictly_harder


def _letter_name(i: int) -> str:
    """A, B, ..., Z, AA, AB, ... for i = 0, 1, ..."""
    name = ""
    i += 1
    while i:
        i, rem = divmod(i - 1, 26)
        name = chr(ord("A") + rem) + name
    return name


def _ceiling(nr_empty: int, frontier: list[Candidate], start_empty: int, start_solutions: int) -> int:
    """The most solutions a candidate with `nr_empty` empty spaces can have
    and still survive: it must beat `start` (>= its solutions if it has more
    empty spaces, strictly fewer if equally many) and not be dominated by
    any frontier member, all of which -- subsets are visited in decreasing
    nr_empty order -- have at least as many empty spaces (strictly more:
    strictly fewer solutions; equally many: a tie is fine). 0 or less means
    nothing can survive."""
    ceiling = start_solutions - (nr_empty == start_empty)
    for member in frontier:
        ceiling = min(ceiling, member.nr_solutions - (member.nr_empty > nr_empty))
    return ceiling


def most_difficult_puzzles(
    start: Puzzle, difficulty: str = FRONTIER_DIFFICULTY, verbose: bool = True
) -> tuple[PuzzleBook, SolutionBook, SolveStatsBook]:
    """Search `start`'s Setup for puzzles that beat it outright.

    Subsets are visited in *decreasing* nr_empty_spaces order (largest
    first -- free to compute, see module docstring), which is what lets
    the search tighten itself as it goes: by the time a given subset is
    reached, every frontier member found so far already has at least as
    many empty spaces as it does (they were found earlier in this same
    descending order), so the tightest of their nr_solutions -- not
    `start`'s -- is the real ceiling worth searching for here. This is
    also why nothing below `start_empty` needs visiting at all once
    reached: sorted descending, everything after it is `< start_empty`
    too. In practice this means the search keeps shrinking itself: once
    some subset achieves an unbeatable nr_solutions (1), every
    smaller-empty subset from then on inherits that same tight ceiling
    instead of `start`'s original, looser one.

    The ceiling passed to a given subset's search is the most solutions a
    candidate with that many empty spaces could have and still survive (see
    _ceiling): every frontier member with strictly more empty spaces
    demands strictly fewer solutions, one with equally many allows a tie.
    Once that reaches 0 -- e.g. a frontier member with a single solution
    and more empty spaces -- nothing is left to find and the subset is
    skipped without any placement enumeration or solving, which is what
    makes the late, low-empty-space subsets nearly free. The realization
    found is still tested against the real (running) frontier with
    _dominates itself, so the ceiling only has to never exclude a genuine
    survivor, not to be exact.

    Returns (puzzles, solutions, stats): a PuzzleBook and the matching
    SolutionBook and SolveStatsBook (from solving.solve_puzzlebook, so the
    timings are the same kind as any other solve's) holding only the
    mutually non-dominated frontier -- not every candidate that merely beat
    `start`. The puzzles are named A, B, C, ... from most to fewest empty
    spaces. Each has `difficulty` set to `difficulty` (default
    constants.FRONTIER_DIFFICULTY -- deliberately not one of
    constants.DIFFICULTY_COLORS' own tiers, since it isn't part of the game;
    plotting.plot_puzzlebook colors it via constants.FRONTIER_COLOR instead)
    and a real Solution with every one of its solved grids (nr_solutions is
    `len(solution.grids)`, the same convention the rest of the codebase
    uses -- e.g. plotting.plot_puzzlebook -- rather
    than an attribute tacked onto the Puzzle).
    """
    setup = start.setup
    start_empty = start.nr_empty_spaces
    start_solutions = sum(1 for _ in start.solve())

    block_idxs = list(setup.blocks.keys())
    # r starts at 1: the empty board (nothing pre-placed) is left out of the
    # frontier on purpose. It has the most empty spaces of all, so nothing
    # could ever dominate it and it would trivially sit on every frontier --
    # but it isn't a puzzle, and its solution count is no "difficulty".
    all_subsets = itertools.chain.from_iterable(
        itertools.combinations(block_idxs, r) for r in range(1, len(block_idxs) + 1)
    )
    # Subsets with fewer empty spaces than `start` can never beat it, so they
    # are dropped up front (this also keeps the progress bar's total honest).
    with_empty = (
        (setup.n_cells - sum(setup.blocks[idx].count for idx in s), s) for s in all_subsets
    )
    subsets = sorted(
        (e_s for e_s in with_empty if e_s[0] >= start_empty),
        key=lambda e_s: e_s[0],
        reverse=True,
    )

    frontier: list[Candidate] = []  # maintained mutually non-dominated throughout
    for nr_empty, subset in tqdm(subsets, desc="Searching block subsets", unit="subset", disable=not verbose):
        ceiling = _ceiling(nr_empty, frontier, start_empty, start_solutions)
        if ceiling < 1:
            continue  # no realization of this subset can survive, so don't even look
        best = _best_realization(setup, subset, ceiling=ceiling)
        if best is None:
            continue  # nothing at or below the current ceiling is achievable for this subset
        nr_solutions, placements = best

        if nr_empty == start_empty and nr_solutions == start_solutions:
            continue  # ties start on both -- doesn't beat it

        puzzle = Puzzle(setup, name=f"candidate {sorted(subset)}")
        for idx, placement_idx in placements.items():
            puzzle.place_unchecked(idx, placement_idx)
        candidate = Candidate(puzzle, nr_empty, nr_solutions)

        if any(_dominates(f, candidate) for f in frontier):
            continue  # the shared ceiling was a safe over-approximation -- this one didn't
        frontier = [f for f in frontier if not _dominates(candidate, f)] + [candidate]

    if not frontier:
        raise ValueError(f"No puzzle on this Setup beats {start.name!r}.")

    frontier.sort(key=lambda f: (-f.nr_empty, f.nr_solutions))
    puzzles = []
    for i, (puzzle, _, _) in enumerate(frontier):
        puzzle.name = _letter_name(i)
        puzzle.difficulty = difficulty
        puzzles.append(puzzle)

    book = PuzzleBook(*puzzles, name=f"harder_than_{start.name}" if start.name else "most_difficult")
    solutions, stats = solve_puzzlebook(book)
    return book, solutions, stats
