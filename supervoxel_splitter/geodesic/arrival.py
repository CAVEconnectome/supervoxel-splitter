"""Geodesic arrival-time computation from two seed sets over a cost grid.

`compute_TA_TB` runs arrival from A seeds and B seeds with backend `"dj3d"`
or `"mcp"`. On POSIX, the two passes run in a 2-worker fork pool when
`parallel=True`; the workers are module-level so they pickle cleanly.
"""

import multiprocessing
import os
from time import perf_counter

import numpy as np
from skimage.graph import MCP_Geometric

try:
    import dijkstra3d as _dj3d
except Exception:
    _dj3d = None

from ..state import ArrivalGrids


def _mcp_arrival(args):
    """Fork-pool worker for the MCP backend. Pure compute; no shared state."""
    cost_ds, sampling_ds, starts = args
    mcp = MCP_Geometric(cost_ds, sampling=sampling_ds)
    T, _ = mcp.find_costs(starts, find_all_ends=False)
    return T


def _dj3d_arrival(args):
    """Fork-pool worker for dijkstra3d. Cost is pre-scaled by mean(sampling_ds)
    in the caller — `distance_field` has no per-axis anisotropy parameter.
    """
    cost_scaled, starts = args
    starts = list(starts)
    if len(starts) == 1:
        T = _dj3d.distance_field(cost_scaled, source=starts[0], connectivity=26)
    else:
        T = _dj3d.distance_field(cost_scaled, source=starts, connectivity=26)
    return np.asarray(T, dtype=np.float64)


def _run_dj3d(cost_ds, sampling_ds, A_sub, B_sub, use_pool):
    """dj3d backend: pre-scale cost, then either fork-pool or sequential."""
    scale = float(np.mean(sampling_ds))
    cost_scaled = (cost_ds * scale).astype(np.float32, copy=False)
    payload = [(cost_scaled, A_sub), (cost_scaled, B_sub)]
    if use_pool:
        with multiprocessing.get_context("fork").Pool(processes=2) as pool:
            return pool.map(_dj3d_arrival, payload)
    return [_dj3d_arrival(p) for p in payload]


def _run_mcp(cost_ds, sampling_ds, A_sub, B_sub, use_pool):
    """MCP backend: skimage MCP_Geometric per side, fork-pool or sequential."""
    payload = [(cost_ds, sampling_ds, A_sub), (cost_ds, sampling_ds, B_sub)]
    if use_pool:
        with multiprocessing.get_context("fork").Pool(processes=2) as pool:
            return pool.map(_mcp_arrival, payload)
    return [_mcp_arrival(p) for p in payload]


def compute_TA_TB(
    cost_ds: np.ndarray,
    sampling_ds: tuple[float, float, float],
    mask_ds: np.ndarray,
    A_sub,
    B_sub,
    *,
    backend: str = "dj3d",
    parallel: bool = True,
) -> ArrivalGrids:
    """Arrival times from A and B seed sets over `cost_ds`; out-of-mask
    voxels are set to `inf`. Returns a frozen `ArrivalGrids`.
    """
    if backend not in ("dj3d", "mcp"):
        raise ValueError(f"backend must be 'dj3d' or 'mcp', got {backend!r}")
    if backend == "dj3d" and _dj3d is None:
        raise RuntimeError("backend='dj3d' but dijkstra3d is not installed")

    t0 = perf_counter()
    use_pool = parallel and os.name == "posix"
    runner = _run_dj3d if backend == "dj3d" else _run_mcp
    TA, TB = runner(cost_ds, sampling_ds, A_sub, B_sub, use_pool)
    TA = np.where(mask_ds, TA, np.inf)
    TB = np.where(mask_ds, TB, np.inf)
    return ArrivalGrids(t_a=TA, t_b=TB, backend=backend, elapsed_s=perf_counter() - t0)
