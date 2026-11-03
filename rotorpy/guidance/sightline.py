from typing import Optional

import numpy as np


def reference_relative_position(
    height_m: float,
    ray_origin_ned: np.ndarray,
    ray_direction_ned: np.ndarray,
    reference_height_m: float = 0.0,
) -> Optional[np.ndarray]:
    """
    Body-to-target vector where a camera ray meets the plane of the target reference.

    `ray_origin_ned` is the camera position relative to the body origin. The
    body is `height_m` above the ground and the tracked reference point is
    `reference_height_m` above it (zero for the ground contact point). Returns
    None when the ray does not descend onto that plane.
    """
    origin = np.asarray(ray_origin_ned, dtype=float)
    direction = np.asarray(ray_direction_ned, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return None
    unit = direction / norm
    # NED down is positive, so the camera is height_m - origin_down above the ground.
    drop = float(height_m) - float(origin[2]) - float(reference_height_m)
    if drop <= 0.0 or unit[2] <= 0.0:
        return None
    return origin + (drop / unit[2]) * unit


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
