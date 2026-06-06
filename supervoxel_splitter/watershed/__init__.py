"""Seeded-watershed splitter. Lightweight technique using
`skimage.segmentation.watershed` on the distance transform.
"""

from .splitter import WatershedSplitter

__all__ = ["WatershedSplitter"]
