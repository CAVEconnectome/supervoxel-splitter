"""Shared pytest fixtures."""

import numpy as np
import pytest


@pytest.fixture
def dumbbell():
    """Two 10×10×10 cubes joined by a 20×2×2 neck along the X axis.

    Returns `(mask_xyz, src_xyz, snk_xyz)` with one seed inside each cube.
    Axis lengths are deliberately distinct (50, 30, 20) so axis-order
    confusion fails loudly.
    """
    vol = np.zeros((50, 30, 20), dtype=bool)
    vol[5:15, 10:20, 5:15] = True  # left cube
    vol[35:45, 10:20, 5:15] = True  # right cube
    vol[15:35, 14:16, 9:11] = True  # thin neck
    src_xyz = np.array([[10, 15, 10]], dtype=int)
    snk_xyz = np.array([[40, 15, 10]], dtype=int)
    return vol, src_xyz, snk_xyz
