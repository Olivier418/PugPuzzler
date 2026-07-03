import numpy as np
from scipy.ndimage import label
from partitioning import get_all_partitions
from itertools import chain

class Solver:
    def __init__(self, grid, placements, block_counts, label_structure=None):
        self.grid = grid
        self.label_structure = label_structure
        self.placements = placements
        self.block_counts = block_counts
        self.preplaced = set()          # <-- track explicitly

    def initialize_grid(self, grid):
        unknown = {idx for idx in np.unique(grid)
                   if idx != -1 and idx not in self.placements}
        assert not unknown, f"unknown block index(es): {unknown}"

        for idx in np.unique(grid):
            if idx == -1:
                continue
            mask_indices = np.flatnonzero((grid == idx).ravel())
            assert len(mask_indices) == self.block_counts[idx], \
                f"block {idx}: {len(mask_indices)} cells given, expected {self.block_counts[idx]}"

            candidates = self.placements[idx]
            if not np.any(np.all(candidates == mask_indices[np.newaxis, :], axis=1)):
                raise ValueError(
                    f"block {idx}: given cells don't match any valid "
                    f"placement (wrong shape, disconnected, or out of bounds)"
                )
            self.preplaced.add(idx)

        self.grid = grid

   
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

    def _prune_placement_masks(self, placement_masks, available_spots):
        unavailable_flat = (~available_spots).ravel()
        pruned_masks = {}

        for block_idx, block_mask in placement_masks.items():
            pruned_mask = self._filter_block_mask(block_idx, block_mask, unavailable_flat)
            if pruned_mask is None:
                return None
            pruned_masks[block_idx] = pruned_mask

        return pruned_masks

    def solve(self, mode: int = 0, seed: int = None):
        if seed is not None:
            np.random.seed(seed)
            for arr in self.placements.values():
                np.random.shuffle(arr)

        def finish(on_complete):
            if on_complete is not None:
                yield from on_complete()
            else:
                yield self.grid.copy()

        def rec(placement_masks, available_spots, on_complete=None):
            if not placement_masks:
                yield from finish(on_complete)
                return

            pruned_masks = self._prune_placement_masks(placement_masks, available_spots)
            if pruned_masks is None:
                return

            if mode in (1, 2):
                labels, num_components = label(available_spots, structure=self.label_structure)
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

                self.grid.flat[placement] = next_block_idx
                available_spots.flat[placement] = False

                yield from rec(pruned_masks, available_spots, on_complete)

                self.grid.flat[placement] = -1
                available_spots.flat[placement] = True

        initial_placement_masks = {
            idx: np.ones(self.placements[idx].shape[0], dtype=bool)
            for idx in self.block_counts if idx not in self.preplaced
        }
        initial_available = (self.grid == -1)
        yield from rec(initial_placement_masks, initial_available)