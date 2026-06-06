"""Tests for the non-default reference splitters: WatershedSplitter and
NoopSplitter. Both are scoped to validating the Protocol shape and a
basic sanity behavior, not production parity with geodesic.
"""

import numpy as np

from supervoxel_splitter import SINK, SOURCE, NoopSplitter, WatershedSplitter


def test_watershed_each_seed_lands_on_its_own_side(dumbbell):
    """Seeded watershed on the negated EDT must keep the seeds on their
    sides — the minimum proof that markers + mask are wired correctly.
    """
    mask, src, snk = dumbbell
    r = WatershedSplitter().split(mask, src, snk)
    assert r.labels[10, 15, 10] == SOURCE
    assert r.labels[40, 15, 10] == SINK


def test_noop_splitter_marks_every_foreground_voxel_as_source(dumbbell):
    """NoopSplitter is the plugin-author reference. Sanity-check that the
    `Splitter` contract is met and the result is the documented all-SOURCE.
    """
    mask, src, snk = dumbbell
    r = NoopSplitter().split(mask, src, snk)
    np.testing.assert_array_equal(r.labels.astype(bool), mask)
    assert int((r.labels == SOURCE).sum()) == int(mask.sum())
    assert int((r.labels == SINK).sum()) == 0
