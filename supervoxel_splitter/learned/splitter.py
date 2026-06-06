"""Reference no-op splitter for plugin authors.

Marks every foreground voxel as SOURCE — no actual splitting. Exists to
document the `Splitter` Protocol shape for downstream packages that ship
a learned splitter (e.g. one driven by a segmentation network).
"""

from time import perf_counter

import numpy as np

from .._utils import (
    from_internal_zyx_volume,
    seeds_from_zyx,
    seeds_to_zyx,
    to_internal_zyx_volume,
)
from ..api import SOURCE
from ..state import SplitResult


class NoopSplitter:
    """No-op reference splitter. Returns SOURCE for the whole foreground,
    SINK for nothing. Useful as a smoke test for callers and as a starting
    template for a real learned implementation.
    """

    def __init__(self, *, model_path: str | None = None):
        self.model_path = model_path

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
        out = np.zeros_like(sv_zyx, dtype=np.uint8)
        out[sv_zyx] = SOURCE
        return SplitResult(
            labels=from_internal_zyx_volume(out, vol_order),
            side_of_label={SOURCE: SOURCE},
            snapped_sources=seeds_from_zyx(seeds_to_zyx(sources, seed_order), seed_order),
            snapped_sinks=seeds_from_zyx(seeds_to_zyx(sinks, seed_order), seed_order),
            diagnostics={"total_elapsed_s": perf_counter() - t0},
        )
