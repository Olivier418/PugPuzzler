from __future__ import annotations

import copy
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


# Ceiling on Solver._region_symmetry_cache (see _region_symmetries). Entries
# are tiny (one bool mask key + a handful of index arrays), but the number of
# distinct open regions an emptier board produces is effectively unbounded.
_SYM_CACHE_MAX = 200_000


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

        # The unshuffled starting point _shuffle_placements always works
        # from, so a seed means the same thing however many times solve()
        # is called on this Solver.
        self._base_setup = self.game.setup
        self._base_chosen_placement_idx = dict(self.game.chosen_placement_idx)

        # Geometry-only, independent of placement row order -- safe to
        # read once here rather than rebuild per solve() call like
        # _placement_lookup below (which does depend on that order).
        self._board_symmetries = self.game.setup.board_symmetries

        # Mode 4's node-local symmetries can't be precomputed the way
        # board_symmetries can, but they're still pure board geometry --
        # so the cache lives here rather than being reset per solve()
        # call, and survives seed shuffles unchanged.
        self._region_symmetry_cache = {}

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

    def _build_placement_lookup(self):
        """Per block, cell-set -> placement index: the inverse of
        self.placements. Lets a symmetry that is only known at search
        time (mode 4's are node-local, so nothing can be baked in ahead
        of time) name the placement it maps a given placement onto.
        Built fresh per solve() call, same reason as
        _build_cell_to_placements: it indexes placements by row position,
        which a seed shuffle changes."""
        return {
            block_idx: {
                np.sort(row).tobytes(): idx for idx, row in enumerate(placements)
            }
            for block_idx, placements in self.placements.items()
        }

    def _placement_image(self, block_idx, placement_idx: int, image: np.ndarray) -> int:
        """The placement of the same block that `image` maps placement
        `placement_idx` onto.

        Guaranteed to hit: a lattice isometry carries a valid placement
        to an isometric copy of the same block, and every image we ever
        apply keeps that copy inside the region it came from -- hence on
        the board -- while Setup._compute_placement_indices enumerates
        every on-board isometric copy of every block."""
        cells = self.placements[block_idx][placement_idx]
        return self._placement_lookup[block_idx][np.sort(image[cells]).tobytes()]

    def _region_symmetries(self, available_spots: np.ndarray) -> list[np.ndarray]:
        """Setup.region_symmetries of this node's still-open region,
        memoized on the region mask. The same leftover hole recurs
        constantly across sibling branches (different blocks decided
        earlier, same shape left over), and the point-group sweep is
        essentially the whole per-node cost mode 4 adds over mode 2.

        Capped rather than evicting: on an emptier board the number of
        distinct open regions is effectively unbounded, and a stale
        entry is never wrong (this is pure board geometry), so once the
        cache is full it just stops growing."""
        key = available_spots.tobytes()
        images = self._region_symmetry_cache.get(key)
        if images is None:
            images = self.game.setup.region_symmetries(available_spots)
            if len(self._region_symmetry_cache) < _SYM_CACHE_MAX:
                self._region_symmetry_cache[key] = images
        return images

    def _orbits(self, block_idx, candidates: list[int], images: list[np.ndarray]):
        """Group `candidates` (placement indices of block_idx, all lying
        inside the symmetric region) into orbits under `images`, yielding
        one `(representative, {other_member: image_mapping_rep_onto_it})`
        per orbit.

        `candidates` is ascending, so the first unvisited one is already
        its orbit's minimum -- and since `images` forms a group, its
        orbit under those images is the whole orbit, so the image that
        takes it to each other member falls out of the same sweep."""
        visited = set()
        for p in candidates:
            if p in visited:
                continue
            orbit = {}
            for image in images:
                orbit.setdefault(self._placement_image(block_idx, p, image), image)
            visited |= orbit.keys()
            yield p, {other: img for other, img in orbit.items() if other != p}

    def _active_symmetries(self, available_spots: np.ndarray) -> list[int]:
        """Indices into self._board_symmetries of the symmetries that fix
        every currently-decided cell (preplaced, or placed earlier in
        this branch) pointwise -- i.e. that only permute the still-open
        region. Applying one of these to a completed subtree's solutions
        is then guaranteed to leave every earlier placement untouched and
        just relabel the part of the board still being decided. Always
        includes the identity."""
        unavailable = np.flatnonzero(~available_spots)
        return [
            k for k, image in enumerate(self._board_symmetries)
            if np.array_equal(image[unavailable], unavailable)
        ]

    def _apply_symmetry(self, solution, image: np.ndarray):
        """A completed State/Puzzle snapshot, transformed by `image` --
        the mirror/rotation of `solution` that modes 3/4 skipped
        searching for directly, since the symmetry already guarantees
        it's a valid completion once `solution` is.

        Transforming the *whole* grid is safe even though the symmetry is
        only about the region open at one node: every image used is the
        identity outside that region by construction, so preplaced
        blocks, ancestor-placed blocks and sibling components all come
        back out untouched -- including their placement indices, which
        the lookup below resolves to themselves."""
        transformed = solution.copy()
        transformed.grid = np.empty_like(solution.grid)
        transformed.grid[image] = solution.grid
        transformed.chosen_placement_idx = {
            idx: (self._placement_image(idx, p, image) if p != UNPLACED else UNPLACED)
            for idx, p in solution.chosen_placement_idx.items()
        }
        return transformed

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

    def _shuffle_placements(self, seed: int):
        """Randomise the order placements are tried in (what a benchmark
        seed varies), without disturbing anything outside this solver.

        The shuffle can't be done in place: self.game.setup is shared by
        reference with every other Puzzle in the book -- that's the whole
        point of Setup -- so permuting its placement rows would silently
        invalidate the placement indices they have already recorded. So
        the solver gets a private shallow copy of the Setup carrying its
        own reordered placements dict, and this game's own already-placed
        blocks (a Puzzle's preplaced letters) are re-pointed at the rows
        they moved to.

        Shuffling an index array rather than the placement rows directly
        draws the same random stream, so a given seed still produces the
        same ordering it did before."""
        np.random.seed(seed)
        # Always shuffle the original order, not whatever a previous
        # solve() call on this same Solver left behind, so a given seed
        # means the same thing on every call.
        base = self._base_setup
        placements = base.placements
        order = {}
        for block_idx, arr in placements.items():
            perm = np.arange(arr.shape[0])
            np.random.shuffle(perm)
            order[block_idx] = perm

        setup = copy.copy(base)
        setup.placements = {
            block_idx: placements[block_idx][perm] for block_idx, perm in order.items()
        }
        self.game.setup = setup
        self.placements = setup.placements

        # new_rows[i] = old_rows[perm[i]], so the block's original row p now
        # lives at argsort(perm)[p].
        inverse = {block_idx: np.argsort(perm) for block_idx, perm in order.items()}
        self.game.chosen_placement_idx = {
            block_idx: (
                int(inverse[block_idx][p]) if p != UNPLACED else UNPLACED
            )
            for block_idx, p in self._base_chosen_placement_idx.items()
        }

    def solve(self, mode: int = 2, seed: int = None):
        if seed is not None:
            self._shuffle_placements(seed)

        self._cell_to_placements = self._build_cell_to_placements()
        self._placement_lookup = (
            self._build_placement_lookup() if mode in (3, 4) else None
        )

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

            if mode in (1, 2, 3, 4):
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

                    if mode in (2, 3, 4):
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

            # A forced move has nothing to collapse -- one candidate is its
            # own orbit whatever the region's symmetry is -- so don't go
            # looking for symmetries that can't be used. Worth a line
            # because min(counts) picks the most constrained block, which
            # makes forced moves the common case.
            single_candidate = counts[next_block_idx] == 1

            # Mode 3 filters the precomputed whole-board symmetries down to
            # those that leave every already-decided cell where it is; mode
            # 4 instead derives the still-open region's own symmetry group
            # from scratch at this node. The latter is a strict
            # generalization -- any whole-board symmetry fixing the decided
            # cells necessarily maps the open region onto itself, so it
            # turns up in the region sweep too -- and it additionally finds
            # symmetric leftover pockets that no rigid motion of the whole
            # board could ever produce.
            if single_candidate or mode not in (3, 4):
                images = None
            elif mode == 3:
                active_k = self._active_symmetries(available_spots)
                images = [self._board_symmetries[k] for k in active_k]
            else:
                images = self._region_symmetries(available_spots)

            if images is not None and len(images) > 1:
                # The still-open region admits a nontrivial symmetry: group
                # next_block_idx's candidate placements into orbits under it
                # and only actually recurse into one representative per
                # orbit. Every other member's solutions are then just a
                # cheap grid transform of the representative's -- see
                # _apply_symmetry -- since a symmetry of the open region
                # maps one valid completion of it to another.
                #
                # Nesting takes care of itself: a deeper node's region group
                # need not be a subgroup of this one's (that's the point of
                # mode 4), but the recursive call is still required to yield
                # every completion, and transforming a whole completed grid
                # by an image that's the identity outside this region can't
                # disturb what any ancestor already decided. The symmetries
                # found at successive depths therefore just multiply out.
                candidates = np.flatnonzero(removed_mask).tolist()

                for rep, image_for in self._orbits(next_block_idx, candidates, images):
                    placement = self.placements[next_block_idx][rep]

                    self.game.place_unchecked(next_block_idx, rep)
                    available_spots.flat[placement] = False

                    for solution in rec(pruned_masks, available_spots, on_complete, newly_unavailable=placement):
                        yield solution
                        for image in image_for.values():
                            yield self._apply_symmetry(solution, image)

                    self.game.remove_unchecked(next_block_idx)
                    available_spots.flat[placement] = True
                return

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