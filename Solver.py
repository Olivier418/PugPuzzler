from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import label
from partitioning import get_all_partitions
from itertools import chain
from constants import EMPTY, UNPLACED

if TYPE_CHECKING:
    # Only for the type hint below -- never imported at runtime, so this
    # doesn't create a circular import with classes.py (which imports
    # Solver at the top of the file).
    from classes import Game


class Solver:
    """Brute-force / partition-pruned solver for a Game (or Puzzle).

    Only reads Game's public surface (board, blocks, placements,
    chosen_placement_idx, grid, place(), remove(), copy()) -- no
    knowledge of Puzzle's letter-grid loading or any other
    subclass-specific behavior is required.
    """

    def __init__(self, game: "Game"):
        # A private working copy: place()/remove() mutate grid and
        # chosen_placement_idx in lockstep as the search backtracks, but
        # there's no reason that churn should be visible on (or tied to)
        # the caller's own game. self.game.copy() is cheap -- setup
        # (board/blocks/placements) is shared by reference, only the
        # small per-instance grid + chosen_placement_idx dict are copied.
        self.game = game.copy()

        # Cached once per board/lattice; reused as scipy's connectivity
        # structure for splitting the empty region into components.
        self.label_structure = self.game.board.lattice.neighbor_structure

        # game.grid/available_spots are compact (n_cells,), but
        # scipy.ndimage.label needs a real N-D array to find geometric
        # neighbors. Reused every call below as scratch space to scatter
        # a compact mask into, label, then gather back down -- avoids
        # reallocating a full board-shaped array on every node.
        self._label_scratch = np.empty(self.game.board.cells.shape, dtype=bool)

        self.placements = self.game.placements
        self.block_counts = {idx: block.count for idx, block in self.game.blocks.items()}

        # Blocks the game already considers placed (e.g. letters baked
        # into a Puzzle's starting grid) are fixed and excluded from the
        # search entirely.
        self.preplaced = {
            idx for idx, placement_idx in self.game.chosen_placement_idx.items()
            if placement_idx != UNPLACED
        }

    def _filter_block_mask(self, block_idx, block_mask, unavailable_flat):
        active_indices = np.flatnonzero(block_mask)
        if active_indices.size == 0:
            return None

        placements = self.placements[block_idx][active_indices]      # (M, count)
        valid = ~np.any(unavailable_flat[placements], axis=1)         # gather, not broadcast-compare
        kept_indices = active_indices[valid]

        if kept_indices.size == 0:
            return None

        pruned_mask = np.zeros(block_mask.shape, dtype=bool)
        pruned_mask[kept_indices] = True
        return pruned_mask

    def _label_available(self, available_spots: np.ndarray) -> tuple[np.ndarray, int]:
        """available_spots is compact (n_cells,); scipy.ndimage.label
        needs a real N-D array to find geometric neighbors, so scatter
        into the full board-shaped scratch array, label, then gather the
        per-cell labels back down to compact space. Isolated here as the
        one place that needs the full board shape at all."""
        setup = self.game.setup
        self._label_scratch[:] = False
        self._label_scratch.ravel()[setup.compact_to_flat] = available_spots
        labels, num_components = label(self._label_scratch, structure=self.label_structure)
        return labels.ravel()[setup.compact_to_flat], num_components

    def _prune_placement_masks(self, placement_masks, available_spots):
        unavailable_flat = (~available_spots).ravel()
        pruned_masks = {}

        for block_idx, block_mask in placement_masks.items():
            pruned_mask = self._filter_block_mask(block_idx, block_mask, unavailable_flat)
            if pruned_mask is None:
                return None
            pruned_masks[block_idx] = pruned_mask

        return pruned_masks

    def solve(self, mode: int = 2, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            for arr in self.placements.values():
                np.random.shuffle(arr)

        def finish(on_complete):
            if on_complete is not None:
                yield from on_complete()
            else:
                # A full Game snapshot, independent of self.game (which
                # keeps getting mutated as the search backtracks further).
                yield self.game.copy()

        def rec(placement_masks, available_spots, on_complete=None):
            if not placement_masks:
                yield from finish(on_complete)
                return

            pruned_masks = self._prune_placement_masks(placement_masks, available_spots)
            if pruned_masks is None:
                return

            if mode in (1, 2):
                labels, num_components = self._label_available(available_spots)
                if num_components > 1:
                    comp_label_ids = list(range(1, num_components + 1))
                    component_sizes = [int(np.sum(labels == lbl)) for lbl in comp_label_ids]

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
                            (comp_masks, comp_mask), rest = components[0], components[1:]
                            yield from rec(comp_masks, comp_mask, lambda: chain_components(rest, on_complete))

                        for partitioning in chain([first_partition], partitions_gen):
                            components = []
                            for comp_pos, group in enumerate(partitioning):
                                comp_mask = (labels == comp_label_ids[comp_pos])
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
                                    components.append((comp_block_masks, comp_mask))
                                    continue
                                break
                            else:
                                yield from chain_components(components, on_complete)
                        return

            next_block_idx = min(pruned_masks, key=lambda k: np.count_nonzero(pruned_masks[k]))
            removed_mask = pruned_masks.pop(next_block_idx)

            for placement_idx in np.flatnonzero(removed_mask):
                placement = self.placements[next_block_idx][placement_idx]  # (count,) flat indices

                self.game.place_unchecked(next_block_idx, placement_idx)
                available_spots.flat[placement] = False

                yield from rec(pruned_masks, available_spots, on_complete)

                self.game.remove_unchecked(next_block_idx)
                available_spots.flat[placement] = True

        initial_placement_masks = {
            idx: np.ones(self.placements[idx].shape[0], dtype=bool)
            for idx in self.block_counts if idx not in self.preplaced
        }
        initial_available = (self.game.grid == EMPTY)
        yield from rec(initial_placement_masks, initial_available)