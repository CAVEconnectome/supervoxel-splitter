"""Snap seeds onto the nearest True voxel of a mask.

Public entry: `snap_seeds_to_segment`. Internal helpers extract the
mask-boundary voxels, optionally downsample the candidate set, and either
scan the whole mask or grow a seed-local bbox window until it contains
candidates.
"""

from time import perf_counter

import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree

from ._utils import get_logger
from .state import SnapResult

logger = get_logger(__name__)


def _extract_mask_boundary(mask: np.ndarray, erosion_iters: int) -> np.ndarray:
    """Boundary voxels of a 3D bool mask via `iters` rounds of erosion."""
    if erosion_iters < 1:
        return mask.copy()
    structure = np.ones((3, 3, 3), dtype=bool)
    interior = ndi.binary_erosion(
        mask, structure=structure, iterations=erosion_iters, border_value=0
    )
    return mask & (~interior)


def _downsample_points(
    points: np.ndarray,
    mode: str,
    stride: int,
    target: int | None,
    rng: np.random.Generator | None,
) -> np.ndarray:
    """Reduce candidate-point count by stride or uniform random sampling."""
    n = points.shape[0]
    if n == 0:
        return points
    if mode == "stride":
        stride = max(1, int(stride))
        return points[::stride]
    if mode == "random":
        if target is None:
            target = min(n, 50_000)
        target = max(1, int(target))
        if target >= n:
            return points
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.choice(n, size=target, replace=False)
        return points[idx]
    raise ValueError("downsample mode must be 'stride' or 'random'")


def _collect_candidates(
    mask: np.ndarray,
    *,
    mask_order: str,
    use_boundary: bool,
    erosion_iters: int,
    use_bbox: bool,
    seeds_xyz: np.ndarray,
    voxel_size,
    bbox_pad_phys: float | None,
) -> np.ndarray:
    """Return candidate voxel coords in XYZ, either over the full mask or
    over a seed-local bbox that grows until it contains foreground.
    """
    ax_xyz = (2, 1, 0) if mask_order == "zyx" else (0, 1, 2)
    if mask_order == "zyx":
        max_x, max_y, max_z = mask.shape[2] - 1, mask.shape[1] - 1, mask.shape[0] - 1
    else:
        max_x, max_y, max_z = mask.shape[0] - 1, mask.shape[1] - 1, mask.shape[2] - 1

    def _from_window(window_mask: np.ndarray, origin_xyz) -> np.ndarray:
        if use_boundary:
            cand = _extract_mask_boundary(window_mask, erosion_iters)
            if not cand.any():
                cand = window_mask
                logger.debug("boundary empty → fallback to full mask")
        else:
            cand = window_mask
        wc = np.where(cand)
        pts = np.stack([wc[ax_xyz[0]], wc[ax_xyz[1]], wc[ax_xyz[2]]], axis=1)
        return pts + np.asarray(origin_xyz, dtype=pts.dtype)

    if not (use_bbox and seeds_xyz.shape[0] > 0):
        return _from_window(mask, (0, 0, 0))

    # The bbox grows physically (×2 per round) until it contains a candidate
    # or it reaches the full mask. Pad size is correctness-independent.
    vsize_xyz = np.asarray(voxel_size, dtype=np.float64)
    if bbox_pad_phys is None:
        bbox_pad_phys = float(vsize_xyz.max())
    seed_min_xyz = np.floor(seeds_xyz.min(axis=0)).astype(np.int64)
    seed_max_xyz = np.ceil(seeds_xyz.max(axis=0)).astype(np.int64)
    full_max_xyz = np.array([max_x, max_y, max_z], dtype=np.int64)
    pad_phys = float(bbox_pad_phys)
    while True:
        pad_vox = np.ceil(pad_phys / vsize_xyz).astype(np.int64)
        lo_xyz = np.maximum(seed_min_xyz - pad_vox, 0)
        hi_xyz = np.minimum(seed_max_xyz + pad_vox, full_max_xyz)
        lo_mask = lo_xyz[list(ax_xyz)]
        hi_mask = hi_xyz[list(ax_xyz)]
        sl = tuple(slice(int(lo_mask[a]), int(hi_mask[a]) + 1) for a in range(3))
        points_xyz = _from_window(mask[sl], lo_xyz)
        if points_xyz.shape[0] > 0:
            return points_xyz
        if np.all(lo_xyz == 0) and np.all(hi_xyz == full_max_xyz):
            return points_xyz
        pad_phys *= 2.0


def snap_seeds_to_segment(
    seeds_xyz,
    mask: np.ndarray,
    *,
    mask_order: str = "zyx",
    voxel_size=(1.0, 1.0, 1.0),
    use_boundary: bool = True,
    erosion_iters: int = 1,
    downsample: bool = True,
    downsample_mode: str = "stride",
    downsample_stride: int = 2,
    downsample_target: int | None = None,
    use_bbox: bool = False,
    bbox_pad_phys: float | None = None,
    rng: np.random.Generator | None = None,
    leafsize: int = 16,
) -> SnapResult:
    """Snap each seed in `seeds_xyz` to the nearest True voxel of `mask`.

    Returns a `SnapResult` with the snapped XYZ coords, the count of seeds
    that moved, and total elapsed seconds. Candidate scanning honors
    `use_boundary` (boundary voxels only when True) and `use_bbox` (a
    seed-local window that auto-grows until it contains foreground).
    """
    t0 = perf_counter()

    if mask.ndim != 3:
        raise ValueError("mask must be a 3D boolean array")
    if mask.dtype != bool:
        mask = mask.astype(bool)
    if mask_order not in ("zyx", "xyz"):
        raise ValueError("mask_order must be 'zyx' or 'xyz'")

    seeds_xyz = np.asarray(seeds_xyz, dtype=np.float64)
    if seeds_xyz.ndim == 1:
        seeds_xyz = seeds_xyz[None, :]
    if seeds_xyz.shape[1] != 3:
        raise ValueError("seeds_xyz must be shape (N, 3)")

    if mask_order == "zyx":
        max_x, max_y, max_z = mask.shape[2] - 1, mask.shape[1] - 1, mask.shape[0] - 1
    else:
        max_x, max_y, max_z = mask.shape[0] - 1, mask.shape[1] - 1, mask.shape[2] - 1

    candidates_xyz = _collect_candidates(
        mask,
        mask_order=mask_order,
        use_boundary=use_boundary,
        erosion_iters=erosion_iters,
        use_bbox=use_bbox,
        seeds_xyz=seeds_xyz,
        voxel_size=voxel_size,
        bbox_pad_phys=bbox_pad_phys,
    )
    if candidates_xyz.shape[0] == 0:
        raise ValueError("mask (or boundary) contains no True voxels")

    if downsample:
        candidates_xyz = _downsample_points(
            candidates_xyz,
            mode=downsample_mode,
            stride=downsample_stride,
            target=downsample_target,
            rng=rng,
        )

    # Physical-space coords so the cKDTree honors anisotropy.
    scale = np.asarray(voxel_size, dtype=np.float64)
    tree = cKDTree(candidates_xyz * scale[None, :], leafsize=leafsize)
    _, nn_idx = tree.query(seeds_xyz * scale[None, :], k=1, workers=-1)

    snapped = candidates_xyz[nn_idx].astype(np.int64)
    snapped[:, 0] = np.clip(snapped[:, 0], 0, max_x)
    snapped[:, 1] = np.clip(snapped[:, 1], 0, max_y)
    snapped[:, 2] = np.clip(snapped[:, 2], 0, max_z)

    seeds_round = np.round(seeds_xyz).astype(np.int64)
    moved = int(np.any(snapped != seeds_round, axis=1).sum())
    elapsed = perf_counter() - t0
    logger.debug("snap: %d seeds, %d moved, %.3fs", len(seeds_xyz), moved, elapsed)
    return SnapResult(snapped=snapped, moved_count=moved, elapsed_s=elapsed)
