"""End-to-end tests for `GeodesicSplitter` on the dumbbell shape."""

import numpy as np
import pytest

from supervoxel_splitter import SINK, SOURCE, STRAY, GeodesicSplitter
from supervoxel_splitter._utils import cc_label_26

try:
    import dijkstra3d as _dj3d  # noqa: F401

    _HAS_DJ3D = True
except Exception:
    _HAS_DJ3D = False


def _split(mask, src, snk, **kw):
    """Always disable the fork-pool in tests; pytest + fork is fragile."""
    kw.setdefault("parallel", False)
    return GeodesicSplitter(**kw).split(mask, src, snk)


def test_dumbbell_every_foreground_voxel_labeled(dumbbell):
    """No voxel of the mask is left at 0; the partition is exhaustive."""
    mask, src, snk = dumbbell
    r = _split(mask, src, snk)
    labeled = (r.labels == SOURCE) | (r.labels == SINK) | (r.labels == STRAY)
    np.testing.assert_array_equal(labeled, mask)


def test_dumbbell_each_seed_lands_on_its_own_side(dumbbell):
    """The source seed voxel ends up SOURCE; the sink seed voxel ends up SINK."""
    mask, src, snk = dumbbell
    r = _split(mask, src, snk)
    assert r.labels[10, 15, 10] == SOURCE
    assert r.labels[40, 15, 10] == SINK
    assert int((r.labels == SOURCE).sum()) > 100
    assert int((r.labels == SINK).sum()) > 100


def test_dumbbell_each_side_is_single_connected_component(dumbbell):
    """`enforce_single_cc=True` (default) must collapse each label down to
    one CC. If the enforce-cc pass were dropped, we'd get fragments here.
    """
    mask, src, snk = dumbbell
    r = _split(mask, src, snk)
    _, n_src = cc_label_26(r.labels == SOURCE)
    _, n_snk = cc_label_26(r.labels == SINK)
    assert n_src == 1
    assert n_snk == 1


@pytest.mark.skipif(not _HAS_DJ3D, reason="dijkstra3d not installed")
def test_dj3d_and_mcp_both_produce_valid_split(dumbbell):
    """Both backends may differ near the cut surface but each must keep
    the seeds on their assigned sides — the user-facing contract.
    """
    mask, src, snk = dumbbell
    for backend in ("dj3d", "mcp"):
        r = _split(mask, src, snk, backend=backend)
        assert r.labels[10, 15, 10] == SOURCE
        assert r.labels[40, 15, 10] == SINK


def test_diagnostics_carry_backend_and_label_counts(dumbbell):
    """The diagnostics dict is part of the public SplitResult contract;
    callers (PCG, plugin authors) read these fields.
    """
    mask, src, snk = dumbbell
    r = _split(mask, src, snk)
    counts = r.diagnostics["label_counts"]
    assert counts[SOURCE] > 0
    assert counts[SINK] > 0
    assert r.diagnostics["backend"] in ("dj3d", "mcp")
    assert r.diagnostics["full_shape_zyx"] == (20, 30, 50)  # ZYX of xyz=(50,30,20)
