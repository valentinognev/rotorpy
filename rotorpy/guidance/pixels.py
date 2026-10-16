"""Target-pixel detection for camera homing."""

from typing import Optional

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
