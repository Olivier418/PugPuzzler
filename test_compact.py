import numpy as np

from classes import Board, PyramidBoard
from compact import CompactBoard, ComponentTracker


def make_dumbbell():
    # 5x5 grid: two 2x5 blobs (cols 0-1 and cols 3-4) joined only by a
    # single bridge cell at (row=2, col=2). Orthogonal 4-connectivity.
    cells = np.zeros((5, 5), dtype=bool)
    cells[:, 0:2] = True
    cells[:, 3:5] = True
    cells[2, 2] = True
    offset_adjacency = np.zeros((2, 2), dtype=bool)
    return Board(cells, offset_adjacency)


def test_adjacency_matches_expected_neighbor_count():
    board = make_dumbbell()
    cb = CompactBoard(board)
    assert cb.n == 21, cb.n

    # every unit vector for an orthogonal 2D lattice should be the 4
    # axis-aligned neighbors
    uv = {tuple(v) for v in board.lattice.unit_vectors}
    assert uv == {(1, 0), (-1, 0), (0, 1), (0, -1)}, uv

    # the bridge cell (2,2) should have exactly 2 neighbors: (2,1) and (2,3)
    bridge_flat = np.ravel_multi_index((2, 2), board.shape)
    bridge_compact = cb.flat_to_compact[bridge_flat]
    assert len(cb.adjacency[bridge_compact]) == 2, cb.adjacency[bridge_compact]
    print("adjacency test passed")


def test_split_and_restore():
    board = make_dumbbell()
    cb = CompactBoard(board)
    available = np.ones(cb.n, dtype=bool)
    tracker = ComponentTracker(cb, available)

    # initially: one connected component (bridge holds it together)
    assert len(tracker.members) == 1
    (only_id,) = tracker.members.keys()
    assert len(tracker.members[only_id]) == 21

    snapshot_component_id = tracker.component_id.copy()

    bridge_flat = np.ravel_multi_index((2, 2), board.shape)
    bridge_compact = cb.flat_to_compact[bridge_flat]

    new_ids = tracker.remove_cells(np.array([bridge_compact]))
    assert len(new_ids) == 1, new_ids
    assert len(tracker.members) == 2
    sizes = sorted(len(m) for m in tracker.members.values())
    assert sizes == [10, 10], sizes

    tracker.restore()
    assert len(tracker.members) == 1
    assert np.array_equal(tracker.component_id, snapshot_component_id)
    print("split_and_restore test passed")


def test_nested_split_and_restore_in_lifo_order():
    board = make_dumbbell()
    cb = CompactBoard(board)
    available = np.ones(cb.n, dtype=bool)
    tracker = ComponentTracker(cb, available)

    snapshot0 = tracker.component_id.copy()
    snapshot0_members = {k: set(v) for k, v in tracker.members.items()}

    bridge_flat = np.ravel_multi_index((2, 2), board.shape)
    bridge_compact = cb.flat_to_compact[bridge_flat]

    # first removal: splits into two 10-cell blobs
    new_ids_1 = tracker.remove_cells(np.array([bridge_compact]))
    assert len(tracker.members) == 2

    # second removal: cut one of the 2x5 blobs in half by removing an
    # entire middle row within it, e.g. row=2 across cols 0-1
    cut_flat = np.array([
        np.ravel_multi_index((2, 0), board.shape),
        np.ravel_multi_index((2, 1), board.shape),
    ])
    cut_compact = cb.flat_to_compact[cut_flat]
    new_ids_2 = tracker.remove_cells(cut_compact)
    assert len(tracker.members) == 3, len(tracker.members)
    sizes = sorted(len(m) for m in tracker.members.values())
    assert sizes == [4, 4, 10], sizes

    # undo in LIFO order -- must return to *exactly* the original state
    tracker.restore()
    tracker.restore()

    assert np.array_equal(tracker.component_id, snapshot0)
    assert len(tracker.members) == 1
    (only_id,) = tracker.members.keys()
    assert tracker.members[only_id] == snapshot0_members[list(snapshot0_members.keys())[0]]
    print("nested_split_and_restore_in_lifo_order test passed")


def test_pyramid_board_smoke():
    # real board from boards.py: check the whole thing builds and starts
    # as one connected component, and that OOB cells were correctly
    # dropped (compact.n < full bounding box size)
    board = PyramidBoard(width=5, depth=5)
    cb = CompactBoard(board)
    full_bbox_size = int(np.prod(board.shape))
    assert cb.n == int(board.cells.sum())
    assert cb.n < full_bbox_size, (cb.n, full_bbox_size)
    print(f"pyramid: {cb.n} real cells out of {full_bbox_size} bounding-box cells "
          f"({100 * cb.n / full_bbox_size:.1f}%)")

    available = np.ones(cb.n, dtype=bool)
    tracker = ComponentTracker(cb, available)
    assert len(tracker.members) == 1, len(tracker.members)
    print("pyramid_board_smoke test passed")


def test_pyramid_full_board_mask_conversion():
    # Regression test for the SolverV2 bug: a full-board-shaped
    # availability mask (OUTSIDE_BOARD cells included, e.g. straight from
    # `game.grid == EMPTY`) must be narrowed to compact space via
    # compact_to_flat *before* it reaches ComponentTracker -- passing it
    # through unconverted throws an IndexError as soon as any OOB-related
    # index exceeds cb.n.
    board = PyramidBoard(width=5, depth=5)
    cb = CompactBoard(board)

    full_board_mask = board.cells.ravel().copy()  # stand-in for game.grid == EMPTY
    assert full_board_mask.size > cb.n  # sanity: this really is the bigger, wrong-shaped array

    compact_mask = full_board_mask[cb.compact_to_flat]
    assert compact_mask.shape[0] == cb.n
    assert compact_mask.all()  # every real cell should be available

    tracker = ComponentTracker(cb, compact_mask)  # would previously IndexError
    assert len(tracker.members) == 1
    (only_id,) = tracker.members.keys()
    assert len(tracker.members[only_id]) == cb.n
    print("pyramid_full_board_mask_conversion test passed")


if __name__ == "__main__":
    test_adjacency_matches_expected_neighbor_count()
    test_split_and_restore()
    test_nested_split_and_restore_in_lifo_order()
    test_pyramid_board_smoke()
    test_pyramid_full_board_mask_conversion()
    print("\nALL TESTS PASSED")
