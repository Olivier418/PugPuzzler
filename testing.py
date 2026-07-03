from classes import FlatGame
from default_settings import BLOCKS, WIDTH, HEIGHT

import numpy as np
from collections import Counter
import time
import numpy as np
from collections import Counter
from itertools import chain
from scipy.ndimage import label
from partitioning import get_all_partitions
import time


def board_symmetries(w, h):
    """All symmetries of a w x h boolean array that preserve its shape."""
    idx = np.arange(w * h).reshape(w, h)
    syms = [idx, np.rot90(idx, 1), np.rot90(idx, 2), np.rot90(idx, 3),
            np.flip(idx, 0), np.rot90(np.flip(idx, 0), 1),
            np.rot90(np.flip(idx, 0), 2), np.rot90(np.flip(idx, 0), 3)]
    return [s.ravel() for s in syms if s.shape == (w, h)]


def local_symmetry_count(mask):
    """
    mask: 2D boolean array, True = part of this component (rest of array is False,
    the component may be embedded anywhere in a larger grid).
    Returns number of non-identity symmetries that map the component's own
    bounding-box shape onto itself.
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    sub = mask[r0:r1 + 1, c0:c1 + 1]
    w, h = sub.shape
    perms = board_symmetries(w, h)[1:]  # skip identity
    sub_flat = sub.ravel()
    return sum(1 for p in perms if np.array_equal(sub_flat, sub_flat[p]))


def diagnose_symmetry_mode2(game, max_nodes=200_000, print_every=20_000,
                             min_component_size=3):
    solver = game.solver
    label_structure = game._label_structure()
    placements = solver.placements
    block_counts = solver.block_counts
    grid = solver.grid

    total_nodes = 0
    total_components_seen = 0
    total_components_symmetric = 0
    symmetric_component_sizes = Counter()
    stop = False
    t0 = time.time()

    def rec(placement_masks, available_spots, depth):
        nonlocal total_nodes, total_components_seen, total_components_symmetric, stop
        if stop:
            return
        total_nodes += 1

        if total_nodes >= max_nodes:
            stop = True
            return
        if total_nodes % print_every == 0:
            elapsed = time.time() - t0
            print(f"  ...{total_nodes} nodes ({elapsed:.1f}s), "
                  f"{total_components_symmetric}/{total_components_seen} "
                  f"components symmetric so far")

        if not placement_masks:
            return

        avail_flat = (~available_spots).ravel()
        pruned_masks = {}
        for idx, mask in placement_masks.items():
            active = np.flatnonzero(mask)
            if active.size == 0:
                return
            cand = placements[idx][active]
            valid = ~np.any(avail_flat[cand], axis=1)
            kept = active[valid]
            if kept.size == 0:
                return
            pm = np.zeros(mask.shape, dtype=bool)
            pm[kept] = True
            pruned_masks[idx] = pm

        labels, num_components = label(available_spots, structure=label_structure)
        if num_components > 1:
            comp_ids = list(range(1, num_components + 1))
            comp_sizes = [int(np.sum(labels == lbl)) for lbl in comp_ids]

            for lbl, size in zip(comp_ids, comp_sizes):
                total_components_seen += 1
                if size < min_component_size:
                    continue
                comp_mask = (labels == lbl)
                n_sym = local_symmetry_count(comp_mask)
                if n_sym > 0:
                    total_components_symmetric += 1
                    symmetric_component_sizes[size] += 1

            block_indices = list(pruned_masks.keys())
            piece_counts = [block_counts[idx] for idx in block_indices]
            partitions_gen = get_all_partitions(piece_counts, comp_sizes)
            first_partition = next(partitions_gen, None)
            if first_partition is None:
                return

            # Just descend into the first valid partitioning per component,
            # like mode=1 would (we only care about node/hit statistics here).
            for comp_pos, group in enumerate(first_partition):
                if stop:
                    return
                comp_mask = (labels == comp_ids[comp_pos])
                comp_unavail_flat = (~comp_mask).ravel()
                comp_block_masks = {}
                for i in group:
                    b_idx = block_indices[i]
                    active = np.flatnonzero(pruned_masks[b_idx])
                    cand = placements[b_idx][active]
                    valid = ~np.any(comp_unavail_flat[cand], axis=1)
                    kept = active[valid]
                    if kept.size == 0:
                        return
                    pm = np.zeros(pruned_masks[b_idx].shape, dtype=bool)
                    pm[kept] = True
                    comp_block_masks[b_idx] = pm

                next_idx = min(comp_block_masks, key=lambda k: np.count_nonzero(comp_block_masks[k]))
                removed_mask = comp_block_masks.pop(next_idx)
                for p_idx in np.flatnonzero(removed_mask):
                    if stop:
                        return
                    placement = placements[next_idx][p_idx]
                    grid.flat[placement] = next_idx
                    comp_mask_local = comp_mask.copy()
                    comp_mask_local.flat[placement] = False
                    rec(comp_block_masks, comp_mask_local, depth + 1)
                    grid.flat[placement] = -1
            return

        next_idx = min(pruned_masks, key=lambda k: np.count_nonzero(pruned_masks[k]))
        removed_mask = pruned_masks.pop(next_idx)
        for p_idx in np.flatnonzero(removed_mask):
            if stop:
                return
            placement = placements[next_idx][p_idx]
            grid.flat[placement] = next_idx
            available_spots.flat[placement] = False
            rec(pruned_masks, available_spots, depth + 1)
            grid.flat[placement] = -1
            available_spots.flat[placement] = True

    initial_masks = {idx: np.ones(placements[idx].shape[0], dtype=bool)
                      for idx in block_counts if idx not in solver.preplaced}
    initial_available = (grid == -1)

    print(f"Diagnosing mode=2 symmetry for '{game.name}' (shape={game.shape}, "
          f"max_nodes={max_nodes}, min_component_size={min_component_size})")
    rec(initial_masks, initial_available, depth=0)
    elapsed = time.time() - t0

    print()
    print(f"Done in {elapsed:.1f}s.  {'(stopped early)' if stop else '(exhausted)'}")
    print(f"Total nodes visited: {total_nodes}")
    print(f"Components seen (size >= {min_component_size}): {total_components_seen}")
    print(f"Components with local symmetry: {total_components_symmetric} "
          f"({100 * total_components_symmetric / max(total_components_seen, 1):.2f}%)")
    print()
    print("Symmetric component sizes (size: count):")
    for size in sorted(symmetric_component_sizes):
        print(f"  {size:>4} : {symmetric_component_sizes[size]}")

my_game = FlatGame(WIDTH, HEIGHT, BLOCKS, "test")
diagnose_symmetry_mode2(my_game)