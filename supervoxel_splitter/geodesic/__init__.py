"""Geodesic splitter — anisotropic EDT-derived speed field, dijkstra-based
arrival, label-3 stray resolver. The default technique.
"""

from .ridge import RidgeConnectPrep
from .splitter import GeodesicSplitter

__all__ = ["GeodesicSplitter", "RidgeConnectPrep"]
