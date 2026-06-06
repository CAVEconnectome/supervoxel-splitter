"""Verify the public Protocols accept plugin classes by structural typing
alone — no inheritance required. This is the plugin-author contract.
"""

import numpy as np

from supervoxel_splitter import SeedPrep, SplitResult, Splitter, SOURCE


class _CompliantSplitter:
    """Inline class that satisfies the Splitter Protocol structurally."""

    def split(self, mask, sources, sinks, *, voxel_size=(1.0, 1.0, 1.0),
              vol_order="xyz", vox_order="xyz", seed_order="xyz"):
        labels = np.where(mask, SOURCE, 0).astype(np.uint8)
        return SplitResult(
            labels=labels,
            side_of_label={SOURCE: SOURCE},
            snapped_sources=np.asarray(sources),
            snapped_sinks=np.asarray(sinks),
        )


class _IncompleteSplitter:
    """Missing `.split()` — must NOT satisfy the Protocol."""

    def something_else(self):
        return None


class _CompliantPrep:
    def prepare(self, mask, sources, sinks, *, voxel_size=(1.0, 1.0, 1.0),
                vol_order="xyz", vox_order="xyz", seed_order="xyz"):
        return np.asarray(sources), np.asarray(sinks), True, True


class _IncompletePrep:
    pass


def test_splitter_protocol_accepts_compliant_plugin_class():
    assert isinstance(_CompliantSplitter(), Splitter)


def test_splitter_protocol_rejects_class_without_split_method():
    assert not isinstance(_IncompleteSplitter(), Splitter)


def test_seed_prep_protocol_accepts_compliant_plugin_class():
    assert isinstance(_CompliantPrep(), SeedPrep)


def test_seed_prep_protocol_rejects_class_without_prepare_method():
    assert not isinstance(_IncompletePrep(), SeedPrep)
