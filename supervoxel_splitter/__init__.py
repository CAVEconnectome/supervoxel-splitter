"""Pluggable supervoxel-splitting algorithms behind a uniform `Splitter`
protocol. See `api.py` for the protocol shapes and `state.py` for
`SplitResult`.
"""

from .api import (
    SOURCE,
    SINK,
    STRAY,
    Splitter,
    SeedPrep,
)
from .geodesic import GeodesicSplitter, RidgeConnectPrep
from .learned import NoopSplitter
from .state import SplitResult
from .watershed import WatershedSplitter

__all__ = [
    "SOURCE",
    "SINK",
    "STRAY",
    "Splitter",
    "SeedPrep",
    "SplitResult",
    "GeodesicSplitter",
    "RidgeConnectPrep",
    "WatershedSplitter",
    "NoopSplitter",
]
