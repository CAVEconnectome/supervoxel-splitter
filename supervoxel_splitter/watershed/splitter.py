"""Seeded-watershed splitter. Reference implementation that proves the
`Splitter` protocol on a non-geodesic technique. Not production-grade —
no single-CC enforcement, no stray resolver.
"""

from time import perf_counter

import numpy as np
from skimage.segmentation import watershed

from .._snap import snap_seeds_to_segment
from .._utils import (
    NullProfiler,
    Profiler,
    compute_edt,
    from_internal_zyx_volume,
    get_logger,
    seeds_from_zyx,
    seeds_to_zyx,
    to_internal_zyx_volume,
    to_zyx_sampling,
)
from ..api import SINK, SOURCE
from ..state import SplitResult

logger = get_logger(__name__)


def _snap_team(pts_zyx: np.ndarray, mask_zyx: np.ndarray, sampling) -> np.ndarray:
    """Snap a team's seeds onto the foreground mask, ZYX in → ZYX out.

    Watershed markers must be at the interior (low -EDT, the basin floor)
    so flooding labels outward correctly — `use_boundary=False` keeps
    seeds away from the EDT-zero boundary band.
    """
    if pts_zyx.size == 0:
        return np.empty((0, 3), dtype=int)
    pts_xyz = pts_zyx[:, [2, 1, 0]]
    result = snap_seeds_to_segment(
        pts_xyz,
        mask=mask_zyx,
        mask_order="zyx",
        voxel_size=(sampling[2], sampling[1], sampling[0]),
        use_boundary=False,
        downsample=False,
    )
    return result.snapped[:, [2, 1, 0]]


class WatershedSplitter:
    """Seeded watershed on the negated EDT. Basins flow uphill toward seeds
    at the distance-transform peaks, so the cut surface lies on the EDT ridge.
    """

    def __init__(
        self,
        *,
        compactness: float = 1e-3,
        watershed_line: bool = False,
        profiler: Profiler = NullProfiler(),
    ):
        self.compactness = compactness
        self.watershed_line = watershed_line
        self.profiler = profiler

    def split(
        self,
        mask: np.ndarray,
        sources: np.ndarray,
        sinks: np.ndarray,
        *,
        voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        vol_order: str = "xyz",
        vox_order: str = "xyz",
        seed_order: str = "xyz",
    ) -> SplitResult:
        t0 = perf_counter()
        sv_zyx, _ = to_internal_zyx_volume(mask, vol_order)
        sampling = to_zyx_sampling(voxel_size, vox_order)

        with self.profiler.profile("snap"):
            src_zyx = _snap_team(seeds_to_zyx(sources, seed_order), sv_zyx, sampling)
            snk_zyx = _snap_team(seeds_to_zyx(sinks, seed_order), sv_zyx, sampling)

        with self.profiler.profile("edt"):
            dist = compute_edt(sv_zyx, sampling)

        with self.profiler.profile("watershed"):
            markers = np.zeros_like(sv_zyx, dtype=np.int32)
            for z, y, x in src_zyx:
                if sv_zyx[z, y, x]:
                    markers[z, y, x] = SOURCE
            for z, y, x in snk_zyx:
                if sv_zyx[z, y, x]:
                    markers[z, y, x] = SINK
            ws = watershed(
                -dist,
                markers=markers,
                mask=sv_zyx,
                compactness=self.compactness,
                watershed_line=self.watershed_line,
            )

        labels = from_internal_zyx_volume(ws.astype(np.uint8), vol_order)
        return SplitResult(
            labels=labels,
            side_of_label={SOURCE: SOURCE, SINK: SINK},
            snapped_sources=seeds_from_zyx(src_zyx, seed_order),
            snapped_sinks=seeds_from_zyx(snk_zyx, seed_order),
            diagnostics={"total_elapsed_s": perf_counter() - t0},
        )
