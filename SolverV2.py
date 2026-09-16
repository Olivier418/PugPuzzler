"""
SolverV2: a drop-in alternative to Solver.py's connectivity step.

Replaces the per-node `scipy.ndimage.label(available_spots, ...)` call --
which rescans the *entire* board, OUTSIDE_BOARD cells included, every
single time -- with a ComponentTracker that maintains component identity
incrementally: a block placement can only ever split the one component it
was placed inside (never merge, never touch any other component), and
backtracking always undoes the most recent split exactly. That asymmetry
is what makes the rollback cheap without needing general dynamic-graph
machinery.

Everything else -- _filter_block_mask / _prune_placement_masks, the
partitioning search itself, and Game -- is left untouched on purpose:
those already scale with placement/piece counts rather than board size,
so they weren't the thing under suspicion, and leaving them alone limits
the surface area for new bugs.

STATUS: the connectivity core (CompactBoard, ComponentTracker) is unit
tested in test_compact.py against the real lattice.py/boards.py and
passes. This file itself has NOT been run -- it depends on constants.py,
partitioning.py, blocks.py and classes.py, none of which were provided.
Before any timing comparison means anything, this needs to be checked
against the existing Solver for solution-set equivalence on a real
puzzle (see the correctness-check note at the bottom).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from partitioning import get_all_partitions
from itertools import chain
from constants import EMPTY, UNPLACED

from compact import CompactBoard, ComponentTracker

if TYPE_CHECKING:
    from classes import Game


class SolverV2:
    def __init__(self, game: "Game"):
        self.game = game.copy()
        self.placements = self.game.placements
        self.block_counts = {idx: block.count for idx, block in self.game.blocks.items()}

        self.preplaced = {
            idx for idx, placement_idx in self.game.chosen_placement_idx.items()
            if placement_idx != UNPLACED
        }

        # Built once per board, same lifetime as self.placements.
        self.compact = CompactBoard(self.game.board)

    # -- unchanged from Solver.py: still operate on full-board placement
    # arrays, since piece-overlap pruning is already O(placement count),
    # not O(board size), and touching it risks bugs for no benefit. --
    def _filter_block_mask(self, block_idx, block_mask, unavailable_flat):
        active_indices = np.flatnonzero(block_mask)
        if active_indices.size == 0:
            return None

        placements = self.placements[block_idx][active_indices]
        valid = ~np.any(unavailable_flat[placements], axis=1)
        kept_indices = active_indices[valid]

        if kept_indices.size == 0:
            return None

        pruned_mask = np.zeros(block_mask.shape, dtype=bool)
        pruned_mask[kept_indices] = True
        return pruned_mask

    def _prune_placement_masks(self, placement_masks, available_spots):
        unavailable_flat = (~available_spots).ravel()
        pruned_masks = {}

        for block_idx, block_mask in placement_masks.items():
            pruned_mask = self._filter_block_mask(block_idx, block_mask, unavailable_flat)
            if pruned_mask is None:
                return None
            pruned_masks[block_idx] = pruned_mask

        return pruned_masks

    def _component_full_mask(self, compact_members: set) -> np.ndarray:
        """Scatter a compact member set back into a full-board boolean
        mask. Only used at an actual split point (mode 2), and only
        costs O(component size) -- not O(board size) -- even though the
        output array is full-board shaped, which is what
        _filter_block_mask needs downstream.
        """
        mask = np.zeros(self.compact.flat_to_compact.shape[0], dtype=bool)
        mask[self.compact.compact_to_flat[list(compact_members)]] = True
        return mask.reshape(self.game.grid.shape)

    def solve(self, mode: int = 2, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            for arr in self.placements.values():
                np.random.shuffle(arr)

        initial_available = (self.game.grid == EMPTY)
        # initial_available is full-board shaped (OUTSIDE_BOARD cells
        # included); ComponentTracker/CompactBoard index everything in
        # compact space, so it has to be narrowed to just the real cells
        # first -- forgetting this is exactly what broke the pyramid case.
        initial_available_compact = initial_available.ravel()[self.compact.compact_to_flat]
        self.tracker = ComponentTracker(self.compact, initial_available_compact)

        def finish(on_complete):
            if on_complete is not None:
                yield from on_complete()
            else:
                yield self.game.copy()

        def rec(placement_masks, available_spots, component_ids, on_complete=None):
            """component_ids: the tracker's component ids that currently
            make up `available_spots` -- this is the incremental
            replacement for `label(available_spots, ...)`. It's threaded
            as a plain argument (not recomputed by scanning the mask)
            so checking "how many components right now" is
            O(components in scope), never O(board size).
            """
            if not placement_masks:
                yield from finish(on_complete)
                return

            pruned_masks = self._prune_placement_masks(placement_masks, available_spots)
            if pruned_masks is None:
                return

            if mode in (1, 2):
                num_components = len(component_ids)
                if num_components > 1:
                    comp_id_list = sorted(component_ids)
                    component_sizes = [len(self.tracker.members[cid]) for cid in comp_id_list]

                    block_indices = list(pruned_masks.keys())
                    piece_counts = [self.block_counts[idx] for idx in block_indices]

                    partitions_gen = get_all_partitions(piece_counts, component_sizes)
                    first_partition = next(partitions_gen, None)
                    if first_partition is None:
                        return  # Dead end

                    if mode == 2:
                        def chain_components(components, on_complete):
                            if not components:
                                yield from finish(on_complete)
                                return
                            (comp_masks, comp_mask, comp_ids), rest = components[0], components[1:]
                            yield from rec(comp_masks, comp_mask, comp_ids, lambda: chain_components(rest, on_complete))

                        for partitioning in chain([first_partition], partitions_gen):
                            components = []
                            for comp_pos, group in enumerate(partitioning):
                                cid = comp_id_list[comp_pos]
                                comp_members = self.tracker.members[cid]
                                comp_mask = self._component_full_mask(comp_members)
                                comp_unavailable_flat = (~comp_mask).ravel()
                                comp_block_masks = {}

                                for i in group:
                                    b_idx = block_indices[i]
                                    block_mask = self._filter_block_mask(
                                        b_idx,
                                        pruned_masks[b_idx],
                                        comp_unavailable_flat,
                                    )
                                    if block_mask is None:
                                        break
                                    comp_block_masks[b_idx] = block_mask
                                else:
                                    components.append((comp_block_masks, comp_mask, {cid}))
                                    continue
                                break
                            else:
                                yield from chain_components(components, on_complete)
                        return

            next_block_idx = min(pruned_masks, key=lambda k: np.count_nonzero(pruned_masks[k]))
            removed_mask = pruned_masks.pop(next_block_idx)

            for placement_idx in np.flatnonzero(removed_mask):
                placement = self.placements[next_block_idx][placement_idx]  # (count,) flat indices
                compact_placement = self.compact.flat_to_compact[placement]

                self.game.place_unchecked(next_block_idx, placement_idx)
                available_spots.flat[placement] = False
                new_ids = self.tracker.remove_cells(compact_placement)
                # functional update, not a mutation -- the parent's own
                # component_ids is untouched, so the next loop iteration
                # (next placement_idx, i.e. the backtrack sibling) starts
                # fresh from the parent's original set automatically.
                child_component_ids = component_ids | set(new_ids)

                yield from rec(pruned_masks, available_spots, child_component_ids, on_complete)

                self.game.remove_unchecked(next_block_idx)
                available_spots.flat[placement] = True
                self.tracker.restore()

        initial_placement_masks = {
            idx: np.ones(self.placements[idx].shape[0], dtype=bool)
            for idx in self.block_counts if idx not in self.preplaced
        }
        initial_component_ids = frozenset(self.tracker.members.keys())
        yield from rec(initial_placement_masks, initial_available, initial_component_ids)


# ---------------------------------------------------------------------------
# Correctness check to run once the missing files (constants.py,
# partitioning.py, blocks.py, classes.py) are available -- before trusting
# any timing comparison:
#
#   from Solver import Solver
#   from SolverV2 import SolverV2
#
#   sols_old = {sol.grid.tobytes() for sol in Solver(some_game).solve(mode=2, seed=0)}
#   sols_new = {sol.grid.tobytes() for sol in SolverV2(some_game).solve(mode=2, seed=0)}
#   assert sols_old == sols_new
#
# Do this on a small board first (fast enough to enumerate every solution)
# before trying it on anything large enough for timing to matter.
# ---------------------------------------------------------------------------