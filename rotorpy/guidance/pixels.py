"""Target-pixel detection for camera homing."""

from typing import Optional, Tuple

import numpy as np


def _target_mask(image: np.ndarray) -> np.ndarray:
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    return (red > 0.5) & (red > green) & (red > blue)


def target_pixel_count(image: np.ndarray) -> int:
    """Count pixels with red > 0.5 and red strictly greater than green and blue."""
    return int(np.count_nonzero(_target_mask(image)))


def target_centroid(image: np.ndarray) -> Optional[np.ndarray]:
    """Mean (column, row) of target pixels, or None when the count is 0."""
    rows, columns = np.nonzero(_target_mask(image))
    if rows.size == 0:
        return None
    return np.array([np.mean(columns), np.mean(rows)])


def target_square(image: np.ndarray) -> Optional[Tuple[float, float, int]]:
    """Bounding square (center_u, center_v, side), or None when the count is 0."""
    rows, columns = np.nonzero(_target_mask(image))
    if rows.size == 0:
        return None
    col_min = int(columns.min())
    col_max = int(columns.max())
    row_min = int(rows.min())
    row_max = int(rows.max())
    center_u = (col_min + col_max) / 2
    center_v = (row_min + row_max) / 2
    side = max(col_max - col_min + 1, row_max - row_min + 1)
    return (center_u, center_v, side)
