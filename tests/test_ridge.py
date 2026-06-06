"""Tests for `RidgeConnectPrep`. Pins the no-MST early-return path and
the core path-tracing logic that bridges same-team seeds.
"""

import numpy as np

from supervoxel_splitter import RidgeConnectPrep


def test_single_seed_each_team_returns_input_unchanged(dumbbell):
    """With one seed per team, the MST has no edges and the path mask is
    empty — aug must equal the snapped originals.
    """
    mask, src, snk = dumbbell
    aug_a, aug_b, ok_a, ok_b = RidgeConnectPrep().prepare(mask, src, snk)
    assert ok_a and ok_b
    assert len(aug_a) == 1
    assert len(aug_b) == 1


def test_two_seeds_same_team_get_connected_by_path(dumbbell):
    """Two source seeds inside the same cube. Ridge MST adds one edge
    between them; the traced path adds voxels to aug_a. The sink team has
    one seed and stays at one.
    """
    mask, src, snk = dumbbell
    src_two = np.vstack([src, [[12, 17, 12]]])  # second source inside the left cube
    aug_a, aug_b, ok_a, ok_b = RidgeConnectPrep().prepare(mask, src_two, snk)
    assert ok_a and ok_b
    assert len(aug_a) > 2  # 2 originals + intermediate path voxels
    assert len(aug_b) == 1
