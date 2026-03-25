"""Downstream polygon rasterization helpers.

This module contains local rasterization utility used by downstream visualization
scripts.
"""

from __future__ import annotations

import numpy as np
from matplotlib.path import Path as MplPath


def rasterize_polygon(coords: np.ndarray, spatial_size: int = 256) -> np.ndarray:
    """Rasterize polygon vertices into binary occupancy map.

    Args:
        coords: Polygon coordinate array `[N,2]`.
        spatial_size: Output raster size.

    Returns:
        Binary raster image `[S,S]`.
    """
    x = np.linspace(-1, 1, spatial_size)
    y = np.linspace(1, -1, spatial_size)
    x_grid, y_grid = np.meshgrid(x, y)
    points = np.vstack((x_grid.flatten(), y_grid.flatten())).T

    path = MplPath(coords)
    mask = path.contains_points(points)
    return mask.reshape((spatial_size, spatial_size)).astype(np.float32)
