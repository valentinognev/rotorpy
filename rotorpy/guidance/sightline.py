from typing import Optional

import numpy as np


def ground_relative_position(height_m: float, sightline_ned: np.ndarray) -> Optional[np.ndarray]:
    """
    Scale a NED sightline until it meets the ground plane.

    Positive NED z points down. Sightline is a direction, not necessarily a
    unit vector. When height_m > 0 and the unit sightline's down-component is
    positive, return s * unit_sightline with s = height_m / unit_down.
    Otherwise return None.
    """
    sightline = np.asarray(sightline_ned, dtype=float)
    norm = float(np.linalg.norm(sightline))
    if height_m <= 0.0 or norm == 0.0:
        return None
    unit_sightline = sightline / norm
    unit_down = float(unit_sightline[2])
    if unit_down <= 0.0:
        return None
    return (height_m / unit_down) * unit_sightline
