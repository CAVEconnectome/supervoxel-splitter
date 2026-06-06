"""Geodesic supervoxel splitter.

`GeodesicSplitter` implements the `Splitter` protocol. The constructor
captures config; each `.split()` call delegates to a private
`_GeodesicRun` state object that orchestrates the phases (snap → bbox
crop → cost → arrival → label → writeback → enforce → resolve strays →
validate). Frozen-dataclass returns from each phase keep the state
thread explicit.
"""

from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import ball

from .._snap import snap_seeds_to_segment
from .._utils import (
    NullProfiler,
    Profiler,
    compute_edt,
    from_internal_zyx_volume,
    get_logger,
    nonzero_bbox_zyx,
    seeds_from_zyx,
    seeds_to_zyx,
    to_internal_zyx_volume,
    to_zyx_sampling,
    upsample_labels,
)
from .._utils import cc_label_26
from ..api import SINK, SOURCE, STRAY, SeedPrep
from ..state import SplitResult
from .arrival import compute_TA_TB
from .resolve import enforce_single_component, resolve_stray_touching
from .ridge import RidgeConnectPrep

logger = get_logger(__name__)


_DEFAULT_SNAP_KWARGS = {
    "use_boundary": True,
    "erosion_iters": 1,
    "downsample": True,
    "downsample_mode": "random",
    "downsample_target": 50_000,
}


class GeodesicSplitter:
    """Anisotropic EDT-derived speed-field carve with neck-aware slowdown.

    Default `seed_prep=RidgeConnectPrep()` bridges thin necks before the
    carve; pass `seed_prep=None` to skip the prep step.
    """

    def __init__(
        self,
        *,
        halo: int = 1,
        gamma_neck: float = 1.6,
        k_prox: float = 2.0,
        lambda_prox: float = 1.0,
        narrow_band_rel: float = 0.08,
        nb_dilate: int = 1,
        downsample_geodesic: tuple[int, int, int] | None = None,
        allow_third_label: bool = True,
        enforce_single_cc: bool = True,
        backend: str = "dj3d",
        parallel: bool = True,
        snap_kwargs: dict | None = None,
        seed_prep: SeedPrep | None = None,
        raise_if_multi_cc: bool = False,
        profiler: Profiler = NullProfiler(),
    ):
        self.halo = halo
        self.gamma_neck = gamma_neck
        self.k_prox = k_prox
        self.lambda_prox = lambda_prox
        self.narrow_band_rel = narrow_band_rel
        self.nb_dilate = nb_dilate
        self.downsample_geodesic = downsample_geodesic
        self.allow_third_label = allow_third_label
        self.enforce_single_cc = enforce_single_cc
        self.backend = backend
        self.parallel = parallel
        self.snap_kwargs = {**_DEFAULT_SNAP_KWARGS, **(snap_kwargs or {})}
        self.seed_prep = RidgeConnectPrep() if seed_prep is None else seed_prep
        self.raise_if_multi_cc = raise_if_multi_cc
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
        aug_sources_out = None
        aug_sinks_out = None
        srcs_used, snks_used = sources, sinks
        if self.seed_prep is not None:
            with self.profiler.profile("seed_prep"):
                aug_a, aug_b, ok_a, ok_b = self.seed_prep.prepare(
                    mask, sources, sinks,
                    voxel_size=voxel_size, vol_order=vol_order,
                    vox_order=vox_order, seed_order=seed_order,
                )
            if not (ok_a and ok_b):
                raise RuntimeError("seed prep failed for at least one team")
            srcs_used, snks_used = aug_a, aug_b
            aug_sources_out, aug_sinks_out = aug_a, aug_b

        run = _GeodesicRun(
            self, mask, srcs_used, snks_used,
            voxel_size=voxel_size, vol_order=vol_order,
            vox_order=vox_order, seed_order=seed_order,
        )
        return run.execute(aug_sources=aug_sources_out, aug_sinks=aug_sinks_out)


class _GeodesicRun:
    """Per-call state for one `GeodesicSplitter.split()` invocation."""

    def __init__(
        self,
        cfg: GeodesicSplitter,
        mask: np.ndarray,
        sources: np.ndarray,
        sinks: np.ndarray,
        *,
        voxel_size,
        vol_order: str,
        vox_order: str,
        seed_order: str,
    ):
        self.cfg = cfg
        self.vol_order = vol_order
        self.seed_order = seed_order
        self.sv_zyx, _ = to_internal_zyx_volume(mask, vol_order)
        self.sampling = to_zyx_sampling(voxel_size, vox_order)
        self.a_full = seeds_to_zyx(sources, seed_order)
        self.b_full = seeds_to_zyx(sinks, seed_order)
        # populated by phase methods
        self.a_snapped: np.ndarray | None = None
        self.b_snapped: np.ndarray | None = None
        self.crop_origin: tuple[int, int, int] | None = None
        self.sv_crop: np.ndarray | None = None
        self.a_crop: np.ndarray | None = None
        self.b_crop: np.ndarray | None = None
        self.travel_cost: np.ndarray | None = None
        self.diagnostics: dict = {}
        self.stage_elapsed: dict[str, float] = {}

    # ---- phases ----

    def _snap_team(self, pts_zyx: np.ndarray) -> np.ndarray:
        if pts_zyx.size == 0:
            return np.empty((0, 3), dtype=int)
        pts_xyz = pts_zyx[:, [2, 1, 0]]
        result = snap_seeds_to_segment(
            pts_xyz,
            mask=self.sv_zyx,
            mask_order="zyx",
            voxel_size=(self.sampling[2], self.sampling[1], self.sampling[0]),
            **self.cfg.snap_kwargs,
        )
        return result.snapped[:, [2, 1, 0]]

    def _crop_to_foreground(self) -> bool:
        """Tight bbox + halo around the foreground mask; populate sv_crop/a_crop/b_crop.
        Returns False if foreground is empty.
        """
        bb = nonzero_bbox_zyx(self.sv_zyx)
        if bb is None:
            return False
        z0, z1, y0, y1, x0, x1 = bb
        Z, Y, X = self.sv_zyx.shape
        h = self.cfg.halo
        z0h, y0h, x0h = max(z0 - h, 0), max(y0 - h, 0), max(x0 - h, 0)
        z1h, y1h, x1h = min(z1 + h, Z), min(y1 + h, Y), min(x1 + h, X)
        self.crop_origin = (z0h, y0h, x0h)
        self.sv_crop = self.sv_zyx[z0h:z1h, y0h:y1h, x0h:x1h]
        offset = np.array([z0h, y0h, x0h])
        self.a_crop = np.asarray(self.a_snapped, dtype=int) - offset
        self.b_crop = np.asarray(self.b_snapped, dtype=int) - offset
        return True

    def _build_cost(self) -> None:
        """EDT → normalized speed → 1/speed travel cost (full-res crop)."""
        dist = compute_edt(self.sv_crop, self.sampling)
        dn = dist / dist.max() if dist.max() > 0 else dist
        speed = np.clip(dn ** max(self.cfg.gamma_neck, 0.0), 1e-6, 1.0)
        cost = np.full_like(speed, 1e12, dtype=float)
        cost[self.sv_crop] = 1.0 / speed[self.sv_crop]
        self.travel_cost = cost

    def _maybe_downsample(self):
        """Apply `downsample_geodesic` to cost+mask+seeds; fall back to full-res
        if any seed disappears from the DS grid.
        """
        ds = self.cfg.downsample_geodesic
        if ds is None:
            return self.travel_cost, self.sv_crop, self.sampling, ds, \
                [tuple(p) for p in self.a_crop.tolist()], \
                [tuple(p) for p in self.b_crop.tolist()]
        dz, dy, dx = (int(s) for s in ds)
        cost_ds = self.travel_cost[::dz, ::dy, ::dx]
        mask_ds = self.sv_crop[::dz, ::dy, ::dx]
        sampling_ds = (self.sampling[0] * dz, self.sampling[1] * dy, self.sampling[2] * dx)

        def _to_ds(pts):
            pts = (np.asarray(pts, int) // np.array([dz, dy, dx])).astype(int)
            Zs, Ys, Xs = mask_ds.shape
            return [(z, y, x) for z, y, x in pts
                    if 0 <= z < Zs and 0 <= y < Ys and 0 <= x < Xs and mask_ds[z, y, x]]

        a_sub, b_sub = _to_ds(self.a_crop), _to_ds(self.b_crop)
        if len(a_sub) == 0 or len(b_sub) == 0:
            return self.travel_cost, self.sv_crop, self.sampling, None, \
                [tuple(p) for p in self.a_crop.tolist()], \
                [tuple(p) for p in self.b_crop.tolist()]
        return cost_ds, mask_ds, sampling_ds, (dz, dy, dx), a_sub, b_sub

    def _label_on_grid(self, arrival, mask_ds, a_sub, b_sub) -> np.ndarray:
        """Narrow-band + proximity-boosted SOURCE/SINK assignment on the (DS) grid."""
        TA, TB = arrival.t_a, arrival.t_b
        finite = np.isfinite(TA) & np.isfinite(TB) & mask_ds
        denom = TA + TB + 1e-12
        reldiff = np.zeros_like(TA)
        reldiff[finite] = np.abs(TA[finite] - TB[finite]) / denom[finite]
        band = finite & (reldiff <= self.cfg.narrow_band_rel)
        if self.cfg.nb_dilate > 0:
            band = ndi.binary_dilation(band, structure=ball(self.cfg.nb_dilate)) & mask_ds
        if band.sum() < 64:
            band = mask_ds.copy()

        denom_a = 1.0 + self.cfg.k_prox * np.exp(-self.cfg.lambda_prox * np.clip(TB, 0, np.inf))
        denom_b = 1.0 + self.cfg.k_prox * np.exp(-self.cfg.lambda_prox * np.clip(TA, 0, np.inf))
        CA, CB = TA / denom_a, TB / denom_b
        labels = np.zeros_like(mask_ds, dtype=np.uint8)
        labels[(CA <= CB) & band] = SOURCE
        labels[(CB < CA) & band] = SINK
        outer = mask_ds & (labels == 0)
        labels[(TA <= TB) & outer] = SOURCE
        labels[(TB < TA) & outer] = SINK
        for z, y, x in a_sub:
            labels[z, y, x] = SOURCE
        for z, y, x in b_sub:
            labels[z, y, x] = SINK
        return labels

    def _writeback(self, sub_labels_ds: np.ndarray, ds: tuple[int, int, int] | None) -> np.ndarray:
        """Upsample sub-labels to crop shape if needed; write into out_zyx; return out_crop view."""
        if ds is not None:
            sub = upsample_labels(sub_labels_ds, ds, self.sv_crop.shape)
            sub[~self.sv_crop] = 0
            for z, y, x in self.a_crop:
                sub[z, y, x] = SOURCE
            for z, y, x in self.b_crop:
                sub[z, y, x] = SINK
        else:
            sub = sub_labels_ds

        out_zyx = np.zeros_like(self.sv_zyx, dtype=np.uint8)
        z0h, y0h, x0h = self.crop_origin
        out_crop = out_zyx[z0h:z0h + self.sv_crop.shape[0],
                           y0h:y0h + self.sv_crop.shape[1],
                           x0h:x0h + self.sv_crop.shape[2]]
        out_crop[self.sv_crop] = SOURCE
        out_crop[sub == SOURCE] = SOURCE
        out_crop[sub == SINK] = SINK
        self._out_zyx = out_zyx
        return out_crop

    def _enforce_and_resolve(self, out_crop: np.ndarray) -> None:
        """1st enforce_cc pass per label → resolve strays → optional 2nd pass."""
        prof = self.cfg.profiler
        cfg = self.cfg
        if cfg.enforce_single_cc:
            with prof.profile("enforce_cc:1st:source"):
                enforce_single_component(out_crop, SOURCE, self.a_crop, allow_stray=cfg.allow_third_label)
            with prof.profile("enforce_cc:1st:sink"):
                enforce_single_component(out_crop, SINK, self.b_crop, allow_stray=cfg.allow_third_label)

        counts = np.bincount(out_crop.ravel(), minlength=4)
        n_stray = int(counts[STRAY])
        moved_src = moved_snk = 0
        if n_stray:
            with prof.profile("resolve_stray"):
                report = resolve_stray_touching(
                    out_crop,
                    seeds_source=self.a_crop, seeds_sink=self.b_crop,
                    sampling=self.sampling,
                )
                moved_src, moved_snk = report.moved_to_source, report.moved_to_sink

        if (moved_src or moved_snk) and cfg.enforce_single_cc:
            with prof.profile("enforce_cc:2nd:source"):
                enforce_single_component(out_crop, SOURCE, self.a_crop, allow_stray=cfg.allow_third_label)
            with prof.profile("enforce_cc:2nd:sink"):
                enforce_single_component(out_crop, SINK, self.b_crop, allow_stray=cfg.allow_third_label)

    def _validate(self, out_crop: np.ndarray) -> None:
        for lab in (SOURCE, SINK):
            _, ncomp = cc_label_26(out_crop == lab)
            if ncomp > 1:
                msg = f"label {lab} has {ncomp} connected components"
                if self.cfg.raise_if_multi_cc:
                    raise ValueError(msg)
                logger.debug(msg)

    # ---- orchestration ----

    def execute(
        self,
        *,
        aug_sources: np.ndarray | None,
        aug_sinks: np.ndarray | None,
    ) -> SplitResult:
        t0 = perf_counter()
        prof = self.cfg.profiler

        with prof.profile("snap"):
            self.a_snapped = self._snap_team(self.a_full)
            self.b_snapped = self._snap_team(self.b_full)

        # Early-exit: no seeds or no foreground → mark everything SOURCE.
        if self.a_snapped.size == 0 or self.b_snapped.size == 0 or not np.any(self.sv_zyx):
            out = np.zeros_like(self.sv_zyx, dtype=np.uint8)
            out[self.sv_zyx] = SOURCE
            return self._make_result(out, aug_sources, aug_sinks, t0)

        if not self._crop_to_foreground():
            out = np.zeros_like(self.sv_zyx, dtype=np.uint8)
            return self._make_result(out, aug_sources, aug_sinks, t0)

        with prof.profile("cost"):
            self._build_cost()

        cost_ds, mask_ds, sampling_ds, ds, a_sub, b_sub = self._maybe_downsample()

        with prof.profile("arrival"):
            arrival = compute_TA_TB(
                cost_ds, sampling_ds, mask_ds, a_sub, b_sub,
                backend=self.cfg.backend, parallel=self.cfg.parallel,
            )
        self.stage_elapsed["arrival"] = arrival.elapsed_s

        with prof.profile("label"):
            sub_labels_ds = self._label_on_grid(arrival, mask_ds, a_sub, b_sub)

        with prof.profile("writeback"):
            out_crop = self._writeback(sub_labels_ds, ds)

        self._enforce_and_resolve(out_crop)
        self._validate(out_crop)

        return self._make_result(self._out_zyx, aug_sources, aug_sinks, t0, arrival_backend=arrival.backend, ds=ds)

    def _make_result(
        self,
        out_zyx: np.ndarray,
        aug_sources: np.ndarray | None,
        aug_sinks: np.ndarray | None,
        t0: float,
        *,
        arrival_backend: str | None = None,
        ds: tuple[int, int, int] | None = None,
    ) -> SplitResult:
        labels = from_internal_zyx_volume(out_zyx, self.vol_order)
        counts = np.bincount(out_zyx.ravel(), minlength=4)
        diagnostics = {
            "label_counts": {int(SOURCE): int(counts[SOURCE]),
                             int(SINK): int(counts[SINK]),
                             int(STRAY): int(counts[STRAY])},
            "full_shape_zyx": tuple(int(s) for s in self.sv_zyx.shape),
            "fg_bbox_shape_zyx": tuple(int(s) for s in (self.sv_crop.shape if self.sv_crop is not None else (0, 0, 0))),
            "downsample_zyx": tuple(int(s) for s in ds) if ds else None,
            "backend": arrival_backend,
            "total_elapsed_s": perf_counter() - t0,
            "stage_elapsed_s": dict(self.stage_elapsed),
        }
        snapped_src = (seeds_from_zyx(self.a_snapped, self.seed_order)
                       if self.a_snapped is not None else np.empty((0, 3), int))
        snapped_snk = (seeds_from_zyx(self.b_snapped, self.seed_order)
                       if self.b_snapped is not None else np.empty((0, 3), int))
        return SplitResult(
            labels=labels,
            side_of_label={SOURCE: SOURCE, SINK: SINK},
            snapped_sources=snapped_src,
            snapped_sinks=snapped_snk,
            aug_sources=aug_sources,
            aug_sinks=aug_sinks,
            diagnostics=diagnostics,
        )
