"""Tests for `snap_seeds_to_segment`.

Pins the perf-vs-correctness contract (`use_bbox` is invariant),
anisotropy propagation (different voxel sizes change the snap target),
and the empty-mask raise.
"""

import numpy as np
import pytest

from supervoxel_splitter._snap import snap_seeds_to_segment


_BASE_KW = dict(mask_order="xyz", use_boundary=False, downsample=False)


def test_seed_far_outside_mask_snaps_onto_foreground():
    """Seed at the origin, mask far away: snap must land on a True voxel."""
    mask = np.zeros((20, 20, 20), dtype=bool)
    mask[10:15, 10:15, 10:15] = True
    seed = np.array([[0, 0, 0]], dtype=int)
    r = snap_seeds_to_segment(seed, mask=mask, **_BASE_KW)
    x, y, z = r.snapped[0]
    assert mask[x, y, z]
    assert r.moved_count == 1


def test_seed_on_mask_has_zero_moved_count():
    """A seed already on a True voxel must not move."""
    mask = np.zeros((10, 10, 10), dtype=bool)
    mask[5, 5, 5] = True
    seed = np.array([[5, 5, 5]], dtype=int)
    r = snap_seeds_to_segment(seed, mask=mask, **_BASE_KW)
    assert r.moved_count == 0
    np.testing.assert_array_equal(r.snapped, [[5, 5, 5]])


def test_empty_mask_raises():
    """The snap-on-empty contract is a hard ValueError, not silent failure."""
    mask = np.zeros((5, 5, 5), dtype=bool)
    seed = np.array([[0, 0, 0]], dtype=int)
    with pytest.raises(ValueError):
        snap_seeds_to_segment(seed, mask=mask, **_BASE_KW)


def test_use_bbox_results_match_full_scan():
    """`use_bbox=True` is a perf-only knob; output must match the full scan
    voxel-for-voxel. The bbox grows until it contains foreground, so this
    invariant holds regardless of mask density.
    """
    mask = np.zeros((30, 20, 10), dtype=bool)
    mask[5:10, 4:8, 3:6] = True
    mask[20:25, 15:18, 7:9] = True
    seeds = np.array([[7, 6, 4], [22, 16, 7]], dtype=int)
    r_full = snap_seeds_to_segment(seeds, mask=mask, use_bbox=False, **_BASE_KW)
    r_bbox = snap_seeds_to_segment(seeds, mask=mask, use_bbox=True, **_BASE_KW)
    np.testing.assert_array_equal(r_bbox.snapped, r_full.snapped)


def test_anisotropic_voxel_size_changes_snap_target():
    """Two equidistant-in-voxel-space candidates; physically closer one
    wins. Catches "we lost voxel_size on the way to cKDTree" regressions.
    """
    mask = np.zeros((2, 2, 1), dtype=bool)  # xyz shape (X=2, Y=2, Z=1)
    mask[1, 0, 0] = True  # candidate A at xyz=(1, 0, 0)
    mask[0, 1, 0] = True  # candidate B at xyz=(0, 1, 0)
    seed = np.array([[0, 0, 0]], dtype=int)

    # x cheap (1), y expensive (10) → A is physically nearer (1 < 10).
    r1 = snap_seeds_to_segment(seed, mask=mask, voxel_size=(1.0, 10.0, 1.0), **_BASE_KW)
    np.testing.assert_array_equal(r1.snapped, [[1, 0, 0]])

    # Flip: x expensive (10), y cheap (1) → B wins.
    r2 = snap_seeds_to_segment(seed, mask=mask, voxel_size=(10.0, 1.0, 1.0), **_BASE_KW)
    np.testing.assert_array_equal(r2.snapped, [[0, 1, 0]])


def test_snapped_coords_are_inside_volume_bounds():
    """The post-snap clip must keep coords inside the array; a far-out
    seed must not produce out-of-bounds output.
    """
    mask = np.zeros((4, 4, 4), dtype=bool)
    mask[3, 3, 3] = True  # corner voxel
    seed = np.array([[100, 100, 100]], dtype=int)
    r = snap_seeds_to_segment(seed, mask=mask, **_BASE_KW)
    x, y, z = r.snapped[0]
    assert 0 <= x < mask.shape[0]
    assert 0 <= y < mask.shape[1]
    assert 0 <= z < mask.shape[2]
    assert mask[x, y, z]
