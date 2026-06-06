"""Single-CC enforcement + stray-label resolver.

`enforce_single_component` keeps the SV-piece that contains the seeds and
demotes the rest to STRAY. `resolve_stray_touching` reassigns STRAY voxels
to SOURCE or SINK based on a 26-neighbour border vote with a kd-tree
tie-break for components that vote evenly.
"""

from time import perf_counter

import numpy as np
from scipy.spatial import cKDTree

from .._utils import cc_label_26, get_logger, largest_component_id
from ..api import SINK, SOURCE, STRAY
from ..state import EnforceCcReport, Resolve3Report

logger = get_logger(__name__)


def enforce_single_component(
    out_labels: np.ndarray,
    label: int,
    seed_pts: np.ndarray,
    *,
    allow_stray: bool = True,
) -> EnforceCcReport:
    """Keep the CC of `label` that contains a seed; demote the rest to STRAY.

    Returns the number of kept components (1 when seeds agree, 0 if `label`
    is absent) and how many voxels were demoted.
    """
    t0 = perf_counter()
    mask = out_labels == label
    if not np.any(mask):
        return EnforceCcReport(label=label, kept_components=0, moved_to_stray=0, elapsed_s=0.0)

    comp, ncomp = cc_label_26(mask)
    if ncomp <= 1:
        return EnforceCcReport(
            label=label, kept_components=1, moved_to_stray=0, elapsed_s=perf_counter() - t0
        )

    keep_ids: set[int] = set()
    Z, Y, X = out_labels.shape
    for z, y, x in seed_pts:
        if 0 <= z < Z and 0 <= y < Y and 0 <= x < X and out_labels[z, y, x] == label:
            cid = int(comp[z, y, x])
            if cid > 0:
                keep_ids.add(cid)
    if not keep_ids:
        keep_ids = {largest_component_id(comp)}

    lut = np.zeros(ncomp + 1, dtype=np.bool_)
    lut[list(keep_ids)] = True
    bad = (comp > 0) & (~lut[comp])
    moved = int(bad.sum())
    if allow_stray and moved:
        out_labels[bad] = STRAY
    elapsed = perf_counter() - t0
    logger.debug("enforce_cc: label=%d kept=%d moved_to_stray=%d", label, len(keep_ids), moved)
    return EnforceCcReport(
        label=label, kept_components=len(keep_ids), moved_to_stray=moved, elapsed_s=elapsed
    )


_NBR_OFFSETS = np.array(
    [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dz, dy, dx) != (0, 0, 0)],
    dtype=np.int32,
)


class _StrayResolver:
    """Per-call sparse representation of STRAY voxels.

    Holds `l3_coords`, the per-voxel CC index, and the per-CC `assign`
    array so the three phases (border vote → kd-tree tie-break → writeback)
    share state without re-scanning the full volume.
    """

    def __init__(self, out_labels: np.ndarray):
        self.out_labels = out_labels
        l3_mask = out_labels == STRAY
        self.comp3, self.n3 = cc_label_26(l3_mask)
        self.l3_coords = np.argwhere(l3_mask)
        self.comp3_at_l3 = self.comp3[tuple(self.l3_coords.T)]
        self.M = self.l3_coords.shape[0]
        self.assign = np.zeros(self.n3 + 1, dtype=np.int16)
        self.tie_break_count = 0

    def border_vote(self) -> np.ndarray:
        """Per-CC majority of 26-neighbour SOURCE vs SINK; returns undecided CC ids."""
        if self.n3 == 0:
            return np.empty(0, dtype=np.int64)
        nbr_coords = self.l3_coords[:, None, :] + _NBR_OFFSETS[None, :, :]
        shape_arr = np.array(self.out_labels.shape, dtype=np.int64)
        in_bounds = np.all((nbr_coords >= 0) & (nbr_coords < shape_arr), axis=-1)
        flat_nbr = nbr_coords[in_bounds]
        flat_labels = self.out_labels[tuple(flat_nbr.T)]
        voxel_idx = np.broadcast_to(np.arange(self.M, dtype=np.int64)[:, None], in_bounds.shape)[in_bounds]
        has_src = np.bincount(voxel_idx[flat_labels == SOURCE], minlength=self.M) > 0
        has_snk = np.bincount(voxel_idx[flat_labels == SINK], minlength=self.M) > 0
        cnt_src = np.bincount(self.comp3_at_l3[has_src], minlength=self.n3 + 1)
        cnt_snk = np.bincount(self.comp3_at_l3[has_snk], minlength=self.n3 + 1)
        self.assign[cnt_src > cnt_snk] = SOURCE
        self.assign[cnt_snk > cnt_src] = SINK
        return np.where(self.assign[1:] == 0)[0] + 1

    def kdtree_tiebreak(self, undec: np.ndarray, seeds_source, seeds_sink, sampling) -> None:
        """Per-undecided CC, query the closer seed via anisotropy-scaled KD-trees."""
        if undec.size == 0 or seeds_source is None or seeds_sink is None:
            return
        if len(seeds_source) == 0 or len(seeds_sink) == 0:
            return
        sampling_arr = np.asarray(sampling, dtype=float)
        l3_phys = self.l3_coords.astype(float) * sampling_arr
        tree_src = cKDTree(np.asarray(seeds_source, dtype=float) * sampling_arr)
        tree_snk = cKDTree(np.asarray(seeds_sink, dtype=float) * sampling_arr)
        d_src, _ = tree_src.query(l3_phys, k=1, workers=-1)
        d_snk, _ = tree_snk.query(l3_phys, k=1, workers=-1)
        closer_to_snk = d_snk < d_src

        pref_snk = np.bincount(self.comp3_at_l3[closer_to_snk], minlength=self.n3 + 1)
        total = np.bincount(self.comp3_at_l3, minlength=self.n3 + 1)
        tie_ids = undec.astype(int)
        choose_snk = pref_snk[tie_ids] > (total[tie_ids] - pref_snk[tie_ids])
        self.assign[tie_ids[choose_snk]] = SINK
        self.assign[tie_ids[~choose_snk]] = SOURCE
        self.tie_break_count = int(tie_ids.size)

    def writeback(self) -> tuple[int, int]:
        """Scatter per-CC assignment back to `out_labels` at STRAY voxels."""
        if self.n3 == 0:
            return 0, 0
        assign_at_l3 = self.assign[self.comp3_at_l3]
        to_src = np.flatnonzero(assign_at_l3 == SOURCE)
        to_snk = np.flatnonzero(assign_at_l3 == SINK)
        if to_src.size:
            self.out_labels[tuple(self.l3_coords[to_src].T)] = SOURCE
        if to_snk.size:
            self.out_labels[tuple(self.l3_coords[to_snk].T)] = SINK
        return int(to_src.size), int(to_snk.size)


def resolve_stray_touching(
    out_labels: np.ndarray,
    *,
    seeds_source=None,
    seeds_sink=None,
    sampling: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> Resolve3Report:
    """Reassign STRAY voxels to SOURCE or SINK via per-CC border vote +
    anisotropy-scaled kd-tree tie-break for components that vote evenly.
    Mutates `out_labels` in place.
    """
    t0 = perf_counter()
    r = _StrayResolver(out_labels)
    undec = r.border_vote()
    r.kdtree_tiebreak(undec, seeds_source, seeds_sink, sampling)
    moved_src, moved_snk = r.writeback()
    elapsed = perf_counter() - t0
    logger.debug(
        "resolve_stray: n_components=%d → source=%d sink=%d tie_breaks=%d %.3fs",
        r.n3, moved_src, moved_snk, r.tie_break_count, elapsed,
    )
    return Resolve3Report(
        n_components=int(r.n3),
        moved_to_source=moved_src,
        moved_to_sink=moved_snk,
        tie_break_count=r.tie_break_count,
        elapsed_s=elapsed,
    )
