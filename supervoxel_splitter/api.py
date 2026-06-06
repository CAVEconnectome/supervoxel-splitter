"""Public Protocols and label constants. The two output side labels are
SOURCE and SINK; STRAY is the transient label-3 used by some techniques
before the resolver re-assigns it.
"""

from typing import Protocol, runtime_checkable

import numpy as np

from .state import SplitResult

SOURCE = 1
SINK = 2
STRAY = 3


@runtime_checkable
class Splitter(Protocol):
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
    ) -> SplitResult: ...


@runtime_checkable
class SeedPrep(Protocol):
    """Optional pre-step before `Splitter.split()`. Geodesic uses it for
    thin-neck ridge bridging; other techniques may ignore it.
    """

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
    ) -> tuple[np.ndarray, np.ndarray, bool, bool]: ...
