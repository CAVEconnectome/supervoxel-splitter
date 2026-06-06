"""Tests for the cross-cutting helpers in `_utils.py`.

Each test pins a behavior that would silently break a splitter if it
regressed: axis-order swap on output, EDT anisotropy lost, bbox slicing
off-by-one, or CC connectivity dropped from 26 to 6.
"""

import numpy as np

from supervoxel_splitter._utils import (
    cc_label_26,
    compute_edt,
    from_internal_zyx_volume,
    nonzero_bbox_zyx,
    seeds_from_zyx,
    seeds_to_zyx,
    to_internal_zyx_volume,
    upsample_bool,
)


# Distinct dim sizes so axis-order swaps fail loudly.
_VOL_XYZ_SHAPE = (5, 6, 7)


def test_to_internal_and_from_internal_roundtrip_xyz():
    vol = np.arange(np.prod(_VOL_XYZ_SHAPE)).reshape(_VOL_XYZ_SHAPE).astype(np.int32)
    internal, _ = to_internal_zyx_volume(vol, "xyz")
    assert internal.shape == _VOL_XYZ_SHAPE[::-1]  # transposed to ZYX
    roundtrip = from_internal_zyx_volume(internal, "xyz")
    np.testing.assert_array_equal(roundtrip, vol)


def test_to_internal_and_from_internal_roundtrip_zyx_is_identity():
    vol = np.arange(np.prod(_VOL_XYZ_SHAPE)).reshape(_VOL_XYZ_SHAPE).astype(np.int32)
    internal, _ = to_internal_zyx_volume(vol, "zyx")
    np.testing.assert_array_equal(internal, vol)
    roundtrip = from_internal_zyx_volume(internal, "zyx")
    np.testing.assert_array_equal(roundtrip, vol)


def test_seeds_roundtrip_xyz_swaps_axis_order():
    """xyz → internal ZYX → back to xyz reproduces the seed coordinates."""
    seeds_xyz = np.array([[1, 2, 3], [10, 20, 30]])
    zyx = seeds_to_zyx(seeds_xyz, "xyz")
    np.testing.assert_array_equal(zyx, np.array([[3, 2, 1], [30, 20, 10]]))
    back = seeds_from_zyx(zyx, "xyz")
    np.testing.assert_array_equal(back, seeds_xyz)


def test_compute_edt_propagates_anisotropy():
    """A 1-D foreground strip ending at a False voxel: EDT at the far end
    equals (distance-in-voxels × axis sampling). Tests the anisotropy
    kwarg actually reaches the underlying backend.
    """
    mask = np.array([[[True, True, True, True, False]]])  # shape (Z=1, Y=1, X=5)
    iso = compute_edt(mask, sampling_zyx=(1.0, 1.0, 1.0))
    np.testing.assert_allclose(iso[0, 0, 0], 4.0)
    np.testing.assert_allclose(iso[0, 0, 3], 1.0)

    aniso = compute_edt(mask, sampling_zyx=(1.0, 1.0, 10.0))
    np.testing.assert_allclose(aniso[0, 0, 0], 40.0)
    np.testing.assert_allclose(aniso[0, 0, 3], 10.0)


def test_compute_edt_all_zero_mask_returns_all_zero():
    mask = np.zeros((3, 4, 5), dtype=bool)
    out = compute_edt(mask, sampling_zyx=(1.0, 1.0, 1.0))
    np.testing.assert_array_equal(out, np.zeros_like(out))


def test_nonzero_bbox_returns_none_for_empty_volume():
    assert nonzero_bbox_zyx(np.zeros((4, 5, 6), dtype=bool)) is None


def test_nonzero_bbox_is_tight_and_half_open():
    """One voxel at (z=3, y=4, x=5) returns (3, 4, 4, 5, 5, 6) — the
    half-open range callers use for slicing.
    """
    vol = np.zeros((10, 10, 10), dtype=bool)
    vol[3, 4, 5] = True
    bb = nonzero_bbox_zyx(vol)
    assert bb == (3, 4, 4, 5, 5, 6)
    z0, z1, y0, y1, x0, x1 = bb
    np.testing.assert_array_equal(vol[z0:z1, y0:y1, x0:x1], np.array([[[True]]]))


def test_cc_label_26_connects_corner_touching_voxels():
    """Two voxels touching only at a corner (no shared face or edge) are
    one component under 26-connectivity. Catches an accidental drop to
    6-conn, which would split them.
    """
    vol = np.zeros((3, 3, 3), dtype=bool)
    vol[0, 0, 0] = True
    vol[1, 1, 1] = True
    _, n = cc_label_26(vol)
    assert n == 1


def test_cc_label_26_separates_disjoint_blobs():
    vol = np.zeros((5, 5, 5), dtype=bool)
    vol[0, 0, 0] = True
    vol[4, 4, 4] = True  # 26-neighbourhoods don't overlap across this gap
    _, n = cc_label_26(vol)
    assert n == 2


def test_upsample_bool_repeats_per_axis_and_clips_to_target_shape():
    """Catches accidental factor swap (e.g. dz/dy/dx mis-order)."""
    src = np.array([[[True, False]]])  # shape (1, 1, 2)
    up = upsample_bool(src, steps=(2, 3, 4), target_shape=(2, 3, 8))
    assert up.shape == (2, 3, 8)
    # First 4 X-voxels are True (from True column), next 4 are False (from False column).
    np.testing.assert_array_equal(up[:, :, :4], np.ones((2, 3, 4), dtype=bool))
    np.testing.assert_array_equal(up[:, :, 4:], np.zeros((2, 3, 4), dtype=bool))
