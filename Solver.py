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

    def _build_cell_to_placements(self):
        """Reverse index: cell -> placement indices that occupy it, per
        block. Lets a single piece placement invalidate only the (few)
        other placements that actually overlap its cells, instead of
        re-testing every remaining placement against the whole board on
        every recursive call. Built fresh per solve() call (after any
        seed-driven shuffle of self.placements' row order) rather than in
        __init__, since it indexes placements by row position and a
        shuffle would otherwise leave it stale."""
        n_cells = self.game.setup.n_cells
        cell_to_placements = {}
        for block_idx, placements in self.placements.items():
            buckets = [[] for _ in range(n_cells)]
            for p_idx, cells in enumerate(placements):
                for c in cells:
                    buckets[c].append(p_idx)
            cell_to_placements[block_idx] = [
                np.array(b, dtype=np.intp) for b in buckets
            ]
        return cell_to_placements

    def _filter_block_mask(self, block_idx, block_mask, unavailable_flat):
        active_indices = np.flatnonzero(block_mask)
        if active_indices.size == 0:
            return None, 0

        placements = self.placements[block_idx][active_indices]      # (M, count)
        valid = ~np.any(unavailable_flat[placements], axis=1)         # gather, not broadcast-compare
        kept_indices = active_indices[valid]

        if kept_indices.size == 0:
            return None, 0

        pruned_mask = np.zeros(block_mask.shape, dtype=bool)
        pruned_mask[kept_indices] = True
        return pruned_mask, kept_indices.size

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
        """Full re-check of every active placement of every block against
        the whole current availability mask. Only needed the first time a
        mask is built for a given available_spots (initial call, or a
        freshly-split connected component in mode 2) -- see
        _prune_placement_masks_incremental for the hot path."""
        unavailable_flat = (~available_spots).ravel()
        pruned_masks = {}
        counts = {}

        for block_idx, block_mask in placement_masks.items():
            pruned_mask, count = self._filter_block_mask(block_idx, block_mask, unavailable_flat)
            if pruned_mask is None:
                return None, None
            pruned_masks[block_idx] = pruned_mask
            counts[block_idx] = count

        return pruned_masks, counts

    def _prune_placement_masks_incremental(self, placement_masks, newly_unavailable):
        """placement_masks is already fully valid against every cell that
        was unavailable *before* this step -- only newly_unavailable (the
        cells the just-placed piece occupies) are new information. So for
        each block, only the (usually tiny) set of placements that touch
        one of those cells can possibly have become invalid; everything
        else is untouched and doesn't need re-testing."""
        pruned_masks = {}
        counts = {}

        for block_idx, mask in placement_masks.items():
            cell_lists = self._cell_to_placements[block_idx]
            affected = [cell_lists[c] for c in newly_unavailable if cell_lists[c].size]

            if affected:
                idxs = np.concatenate(affected)
                new_mask = mask.copy()
                new_mask[idxs] = False
            else:
                new_mask = mask

            count = int(np.count_nonzero(new_mask))
            if count == 0:
                return None, None
            pruned_masks[block_idx] = new_mask
            counts[block_idx] = count

        return pruned_masks, counts

    def solve(self, mode: int = 2, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            for arr in self.placements.values():
                np.random.shuffle(arr)

        self._cell_to_placements = self._build_cell_to_placements()

        def finish(on_complete):
            if on_complete is not None:
                yield from on_complete()
            else:
                # A full Game snapshot, independent of self.game (which
                # keeps getting mutated as the search backtracks further).
                yield self.game.copy()

        def rec(placement_masks, available_spots, on_complete=None, newly_unavailable=None):
            if not placement_masks:
                yield from finish(on_complete)
                return

            if newly_unavailable is None:
                pruned_masks, counts = self._prune_placement_masks(placement_masks, available_spots)
            else:
                pruned_masks, counts = self._prune_placement_masks_incremental(
                    placement_masks, newly_unavailable
                )
            if pruned_masks is None:
                return

            if mode in (1, 2):
                labels, num_components = self._label_available(available_spots)
                if num_components > 1:
                    comp_label_ids = list(range(1, num_components + 1))
                    component_sizes = np.bincount(labels[labels > 0], minlength=num_components + 1)[1:].tolist()

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
                                    block_mask, _ = self._filter_block_mask(
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

            next_block_idx = min(counts, key=counts.get)
            removed_mask = pruned_masks.pop(next_block_idx)

            for placement_idx in np.flatnonzero(removed_mask):
                placement = self.placements[next_block_idx][placement_idx]  # (count,) flat indices

                self.game.place_unchecked(next_block_idx, placement_idx)
                available_spots.flat[placement] = False

                yield from rec(pruned_masks, available_spots, on_complete, newly_unavailable=placement)

                self.game.remove_unchecked(next_block_idx)
                available_spots.flat[placement] = True

        initial_placement_masks = {
            idx: np.ones(self.placements[idx].shape[0], dtype=bool)
            for idx in self.block_counts if idx not in self.preplaced
        }
        initial_available = (self.game.grid == EMPTY)
        yield from rec(initial_placement_masks, initial_available)