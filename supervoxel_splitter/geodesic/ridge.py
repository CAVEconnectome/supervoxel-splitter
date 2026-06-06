"""Ridge-connect seed preparation.

`RidgeConnectPrep` is a `SeedPrep` implementation that bridges thin necks
between seeds of the same team by building a Prim MST over each team's
seed set (in physical-distance space) and tracing a high-ridge path
between MST-connected pairs on a downsampled cost grid derived from the
EDT. Augmented seed sets are the original seeds plus dilated path voxels.
"""

from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from skimage.graph import MCP_Geometric
from skimage.morphology import ball

from .._snap import snap_seeds_to_segment
from .._utils import (
    NullProfiler,
    Profiler,
    compute_edt,
    get_logger,
    seeds_from_zyx,
    seeds_to_zyx,
    to_internal_zyx_volume,
    to_zyx_sampling,
    upsample_bool,
)

logger = get_logger(__name__)


def _bbox_pad_zyx(points_zyx, shape, pad):
    """Padded ZYX bbox enclosing `points_zyx`, clamped to `shape`."""
    pts = np.asarray(points_zyx, int)
    if pts.size == 0:
        return (0, 0, 0, shape[0], shape[1], shape[2])
    z0, y0, x0 = pts.min(0)
    z1, y1, x1 = pts.max(0) + 1
    z0 = max(0, z0 - pad[0])
    y0 = max(0, y0 - pad[1])
    x0 = max(0, x0 - pad[2])
    z1 = min(shape[0], z1 + pad[0])
    y1 = min(shape[1], y1 + pad[1])
    x1 = min(shape[2], x1 + pad[2])
    return (z0, y0, x0, z1, y1, x1)


def _mst_edges_phys(pts_zyx, sampling):
    """Prim's MST edges over `pts_zyx` in anisotropy-scaled physical space."""
    P = np.asarray(pts_zyx, float)
    if len(P) <= 1:
        return []
    phys = P * np.asarray(sampling, float)[None, :]
    n = len(P)
    in_tree = np.zeros(n, bool)
    in_tree[0] = True
    best = np.sqrt(((phys - phys[0]) ** 2).sum(1))
    best[0] = np.inf
    parent = np.zeros(n, int)
    edges = []
    for _ in range(n - 1):
        i = int(np.argmin(best))
        if not np.isfinite(best[i]):
            break
        edges.append((int(parent[i]), i))
        in_tree[i] = True
        best[i] = np.inf
        di = np.sqrt(((phys - phys[i]) ** 2).sum(1))
        relax = (~in_tree) & (di < best)
        parent[relax] = i
        best[relax] = di[relax]
    return edges


def _path_mask(cost: np.ndarray, sampling, start, end) -> np.ndarray | None:
    """MCP_Geometric path mask from `start` to `end`; `None` if unreachable."""
    mcp = MCP_Geometric(cost, sampling=sampling)
    costs, _ = mcp.find_costs([tuple(start)], find_all_ends=False)
    if not np.isfinite(costs[tuple(end)]):
        return None
    path = np.asarray(mcp.traceback(tuple(end)), int)
    m = np.zeros_like(cost, bool)
    m[tuple(path.T)] = True
    return m


def _cost_from_edt(roi: np.ndarray, sampling, ridge_power: float, eps: float = 1e-6) -> np.ndarray:
    """`1 / (eps + normalized_edt ** ridge_power)` inside the ROI mask; big outside."""
    dist = compute_edt(roi, sampling)
    dn = dist / dist.max() if dist.max() > 0 else dist
    cost = np.full_like(dn, 1e12, dtype=float)
    cost[roi] = 1.0 / (eps + np.clip(dn[roi], 0, 1) ** max(0.0, ridge_power))
    return cost


_DEFAULT_SNAP_KWARGS = {
    "use_boundary": True,
    "erosion_iters": 1,
    "downsample": True,
    "downsample_mode": "random",
    "downsample_target": 50_000,
}


class RidgeConnectPrep:
    """SeedPrep impl: bridge thin necks between same-team seeds via MST-of-seeds
    + MCP path-tracing on an EDT-derived ridge cost. Implements `SeedPrep`.
    """

    def __init__(
        self,
        *,
        ridge_power: float = 2.0,
        roi_pad_zyx: tuple[int, int, int] = (24, 48, 48),
        downsample: tuple[int, int, int] = (2, 2, 1),
        refine_fullres_when_fail: bool = True,
        snap_kwargs: dict | None = None,
        profiler: Profiler = NullProfiler(),
    ):
        self.ridge_power = ridge_power
        self.roi_pad_zyx = tuple(roi_pad_zyx)
        self.downsample = tuple(int(s) for s in downsample)
        self.refine_fullres_when_fail = refine_fullres_when_fail
        self.snap_kwargs = {**_DEFAULT_SNAP_KWARGS, **(snap_kwargs or {})}
        self.profiler = profiler

    def prepare(
        self,
        mask: np.ndarray,
        sources: np.ndarray,
        sinks: np.ndarray,
        *,
        voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        vol_order: str = "xyz",
        vox_order: str = "xyz",
        seed_order: str = "xyz",
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        run = _RidgeRun(
            self, mask, sources, sinks,
            voxel_size=voxel_size, vol_order=vol_order, vox_order=vox_order, seed_order=seed_order,
        )
        return run.execute()


class _RidgeRun:
    """Per-call state for one `RidgeConnectPrep.prepare()` invocation."""

    def __init__(
        self,
        cfg: RidgeConnectPrep,
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
        self.seed_order = seed_order
        self.sv_zyx, _ = to_internal_zyx_volume(mask, vol_order)
        self.sampling = to_zyx_sampling(voxel_size, vox_order)
        self.a_in_zyx = seeds_to_zyx(sources, seed_order)
        self.b_in_zyx = seeds_to_zyx(sinks, seed_order)
        # Per-call mutable state, populated by execute().
        self.roi_origin = None  # (z0, y0, x0)
        self.roi = None
        self.roi_ds = None
        self.sampling_ds = None

    # ---- phases ----

    def _snap_team(self, pts_zyx: np.ndarray) -> np.ndarray:
        """Snap a team's seeds to the foreground mask."""
        if pts_zyx.size == 0:
            return np.empty((0, 3), dtype=int)
        pts_xyz = pts_zyx[:, [2, 1, 0]]
        # snap operates in xyz; voxel_size translates ZYX → XYZ.
        result = snap_seeds_to_segment(
            pts_xyz,
            mask=self.sv_zyx,
            mask_order="zyx",
            voxel_size=(self.sampling[2], self.sampling[1], self.sampling[0]),
            **self.cfg.snap_kwargs,
        )
        return result.snapped[:, [2, 1, 0]]

    def _build_roi(self, all_pts_zyx: np.ndarray) -> None:
        """Crop a padded bbox around `all_pts_zyx` and pre-downsample it."""
        z0, y0, x0, z1, y1, x1 = _bbox_pad_zyx(all_pts_zyx, self.sv_zyx.shape, self.cfg.roi_pad_zyx)
        self.roi_origin = (z0, y0, x0)
        self.roi = self.sv_zyx[z0:z1, y0:y1, x0:x1]
        sz, sy, sx = self.cfg.downsample
        self.roi_ds = self.roi[::sz, ::sy, ::sx] if (sz, sy, sx) != (1, 1, 1) else self.roi
        self.sampling_ds = (
            self.sampling[0] * sz, self.sampling[1] * sy, self.sampling[2] * sx,
        )

    def _to_ds_grid(self, pts_zyx: np.ndarray) -> np.ndarray:
        """Map seeds into the downsampled ROI frame, snapping to the nearest True voxel."""
        if pts_zyx.size == 0:
            return np.empty((0, 3), dtype=int)
        sz, sy, sx = self.cfg.downsample
        local = np.asarray(pts_zyx, int) - np.array(self.roi_origin)
        ds = local / np.array([sz, sy, sx], dtype=float)
        try:
            result = snap_seeds_to_segment(
                ds[:, [2, 1, 0]],
                mask=self.roi_ds,
                mask_order="zyx",
                voxel_size=(self.sampling_ds[2], self.sampling_ds[1], self.sampling_ds[0]),
                use_boundary=False,
                downsample=False,
            )
            return result.snapped[:, [2, 1, 0]].astype(int)
        except ValueError:
            # roi_ds empty / degenerate — nearest-int fallback with mask check.
            approx = np.floor(ds + 0.5).astype(int)
            Z, Y, X = self.roi_ds.shape
            approx[:, 0] = np.clip(approx[:, 0], 0, Z - 1)
            approx[:, 1] = np.clip(approx[:, 1], 0, Y - 1)
            approx[:, 2] = np.clip(approx[:, 2], 0, X - 1)
            valid = [tuple(p) for p in approx if self.roi_ds[tuple(p)]]
            return np.array(valid, dtype=int)

    def _augment_team(
        self,
        pts_ds: np.ndarray,
        cost_ds: np.ndarray,
        cost_fr_lazy,
    ) -> tuple[np.ndarray, bool]:
        """MST-of-seeds + path tracing on cost_ds. Lazy full-res fallback when
        the downsampled path can't find one and `refine_fullres_when_fail`.
        """
        if len(pts_ds) <= 1:
            return np.zeros_like(self.roi_ds, bool), True
        sz, sy, sx = self.cfg.downsample
        pmask = np.zeros_like(self.roi_ds, bool)
        ok = True
        for i, j in _mst_edges_phys(pts_ds, self.sampling_ds):
            m = _path_mask(cost_ds, self.sampling_ds, pts_ds[i], pts_ds[j])
            if m is None and self.cfg.refine_fullres_when_fail:
                cost_fr = cost_fr_lazy()
                s = np.array(pts_ds[i]) * np.array([sz, sy, sx])
                e = np.array(pts_ds[j]) * np.array([sz, sy, sx])
                m_fr = _path_mask(cost_fr, self.sampling, s, e)
                if m_fr is not None:
                    m = m_fr[::sz, ::sy, ::sx]
            if m is None:
                ok = False
            else:
                pmask |= m
        return pmask, ok

    def _assemble_team(self, pts_zyx: np.ndarray, path_ds_mask: np.ndarray) -> np.ndarray:
        """Original seeds + upsampled/dilated path voxels, in seed_order."""
        sz, sy, sx = self.cfg.downsample
        path = upsample_bool(path_ds_mask, (sz, sy, sx), self.roi.shape) & self.roi
        path = ndi.binary_dilation(path, structure=ball(1)) & self.roi
        aug = set(map(tuple, pts_zyx))
        z0, y0, x0 = self.roi_origin
        for z, y, x in zip(*np.nonzero(path)):
            aug.add((z0 + z, y0 + y, x0 + x))
        return seeds_from_zyx(np.array(sorted(aug), int), self.seed_order)

    # ---- orchestration ----

    def execute(self) -> tuple[np.ndarray, np.ndarray, bool, bool]:
        t0 = perf_counter()
        prof = self.cfg.profiler
        with prof.profile("ridge:snap"):
            a_zyx = self._snap_team(self.a_in_zyx)
            b_zyx = self._snap_team(self.b_in_zyx)
        if len(a_zyx) == 0 or len(b_zyx) == 0:
            return (
                seeds_from_zyx(a_zyx, self.seed_order),
                seeds_from_zyx(b_zyx, self.seed_order),
                len(a_zyx) > 0,
                len(b_zyx) > 0,
            )

        self._build_roi(np.vstack([a_zyx, b_zyx]))
        a_ds = self._to_ds_grid(a_zyx)
        b_ds = self._to_ds_grid(b_zyx)
        if len(a_ds) == 0 or len(b_ds) == 0:
            return (
                seeds_from_zyx(a_zyx, self.seed_order),
                seeds_from_zyx(b_zyx, self.seed_order),
                len(a_ds) > 0,
                len(b_ds) > 0,
            )

        cost_ds = _cost_from_edt(self.roi_ds, self.sampling_ds, self.cfg.ridge_power)
        if not np.isfinite(cost_ds[self.roi_ds]).any():
            return (
                seeds_from_zyx(a_zyx, self.seed_order),
                seeds_from_zyx(b_zyx, self.seed_order),
                False, False,
            )

        # Full-res fallback cost is built only if needed.
        _cache = {}

        def _cost_fr():
            if "v" not in _cache:
                _cache["v"] = _cost_from_edt(self.roi, self.sampling, self.cfg.ridge_power)
            return _cache["v"]

        pA, okA = self._augment_team(a_ds, cost_ds, _cost_fr)
        pB, okB = self._augment_team(b_ds, cost_ds, _cost_fr)
        if not (okA and okB):
            return (
                seeds_from_zyx(a_zyx, self.seed_order),
                seeds_from_zyx(b_zyx, self.seed_order),
                okA, okB,
            )

        a_aug = self._assemble_team(a_zyx, pA)
        b_aug = self._assemble_team(b_zyx, pB)
        logger.debug(
            "ridge: +%d voxels for A, +%d for B, %.3fs",
            len(a_aug) - len(self.a_in_zyx), len(b_aug) - len(self.b_in_zyx), perf_counter() - t0,
        )
        return a_aug, b_aug, True, True
