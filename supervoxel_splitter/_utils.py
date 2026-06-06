"""Cross-cutting utilities — logger shim, null profiler, axis-order
conversions, EDT wrapper. Folded into one module to avoid a directory of
nine-line files; split out when a category actually grows.
"""

import logging
from contextlib import contextmanager
from typing import Protocol

import cc3d
import fastremap
import numpy as np
from scipy import ndimage as ndi

try:
    from edt import edt as _edt_fast

    _HAVE_EDT_FAST = True
except Exception:
    _HAVE_EDT_FAST = False


# ---- logging ----------------------------------------------------------

logging.getLogger(__name__.rsplit(".", 1)[0]).addHandler(logging.NullHandler())


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


# ---- profiler ---------------------------------------------------------


class Profiler(Protocol):
    def profile(self, name: str): ...


class NullProfiler:
    """No-op profiler used when the caller doesn't inject one. Mirrors the
    `.profile(name)` contextmanager contract of PCG's HierarchicalProfiler.
    """

    @contextmanager
    def profile(self, name: str):
        yield


# ---- axis orders ------------------------------------------------------


def to_zyx_sampling(voxel_size, vox_order: str) -> tuple[float, float, float]:
    vs = tuple(map(float, voxel_size))
    o = vox_order.lower()
    if o == "xyz":
        return (vs[2], vs[1], vs[0])
    if o == "zyx":
        return vs
    raise ValueError("vox_order must be 'xyz' or 'zyx'")


def to_internal_zyx_volume(vol: np.ndarray, vol_order: str) -> tuple[np.ndarray, bool]:
    o = vol_order.lower()
    if o == "zyx":
        return vol, False
    if o == "xyz":
        return np.transpose(vol, (2, 1, 0)), True
    raise ValueError("vol_order must be 'xyz' or 'zyx'")


def from_internal_zyx_volume(vol_zyx: np.ndarray, vol_order: str) -> np.ndarray:
    o = vol_order.lower()
    if o == "zyx":
        return vol_zyx
    if o == "xyz":
        return np.transpose(vol_zyx, (2, 1, 0))
    raise ValueError("vol_order must be 'xyz' or 'zyx'")


def seeds_to_zyx(seeds, seed_order: str) -> np.ndarray:
    arr = np.asarray(seeds, dtype=float).reshape(-1, 3)
    o = seed_order.lower()
    if o == "xyz":
        arr = arr[:, [2, 1, 0]]
    elif o != "zyx":
        raise ValueError("seed_order must be 'xyz' or 'zyx'")
    return np.round(arr).astype(int)


def seeds_from_zyx(seeds_zyx, seed_order: str) -> np.ndarray:
    arr = np.asarray(seeds_zyx, dtype=int).reshape(-1, 3)
    o = seed_order.lower()
    if o == "xyz":
        return arr[:, [2, 1, 0]]
    if o == "zyx":
        return arr
    raise ValueError("seed_order must be 'xyz' or 'zyx'")


# ---- EDT --------------------------------------------------------------


def compute_edt(mask_zyx: np.ndarray, sampling_zyx) -> np.ndarray:
    """Anisotropic EDT of a 3D bool mask in ZYX. Uses Seung-Lab `edt` when
    importable, falls back to `scipy.ndimage.distance_transform_edt`.
    """
    if _HAVE_EDT_FAST:
        return _edt_fast(mask_zyx.astype(np.uint8, copy=False), anisotropy=sampling_zyx)
    return ndi.distance_transform_edt(mask_zyx, sampling=sampling_zyx)


# ---- bbox + upsample --------------------------------------------------


def nonzero_bbox_zyx(vol: np.ndarray) -> tuple[int, int, int, int, int, int] | None:
    """Tight bbox of nonzero voxels via per-axis `np.any` projections.

    Returns `(z0, z1, y0, y1, x0, x1)` as half-open ranges, or `None` if
    the volume has no nonzero voxels. Bandwidth-bound; avoids the per-True
    coord buffer that `np.argwhere` materializes.
    """
    any_z = np.any(vol, axis=(1, 2))
    any_y = np.any(vol, axis=(0, 2))
    any_x = np.any(vol, axis=(0, 1))
    nz_z = np.flatnonzero(any_z)
    if nz_z.size == 0:
        return None
    nz_y = np.flatnonzero(any_y)
    nz_x = np.flatnonzero(any_x)
    return (
        int(nz_z[0]), int(nz_z[-1]) + 1,
        int(nz_y[0]), int(nz_y[-1]) + 1,
        int(nz_x[0]), int(nz_x[-1]) + 1,
    )


def upsample_bool(mask_ds: np.ndarray, steps, target_shape) -> np.ndarray:
    up = mask_ds.repeat(steps[0], 0).repeat(steps[1], 1).repeat(steps[2], 2)
    return up[: target_shape[0], : target_shape[1], : target_shape[2]]


def upsample_labels(lbl_ds: np.ndarray, steps, target_shape) -> np.ndarray:
    up = lbl_ds.repeat(steps[0], 0).repeat(steps[1], 1).repeat(steps[2], 2)
    return up[: target_shape[0], : target_shape[1], : target_shape[2]]


# ---- connected components --------------------------------------------


def cc_label_26(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """26-connected components of a 3D bool mask. Returns (labels, n_components)."""
    return cc3d.connected_components(
        mask, connectivity=26, return_N=True, binary_image=True
    )


def largest_component_id(lbl: np.ndarray) -> int:
    """Label id (>=1) of the largest non-background component in `lbl`."""
    u, counts = fastremap.unique(lbl, return_counts=True)
    if u.size == 0:
        return 0
    bg = np.where(u == 0)[0]
    if bg.size:
        counts[bg[0]] = 0
    return int(u[np.argmax(counts)])
