from typing import Optional

import numpy as np


def pn_acceleration(
    nav_constant: float,
    velocity_ned: np.ndarray,
    sightline_ned: np.ndarray,
    previous_sightline_ned: Optional[np.ndarray],
    dt: float,
) -> np.ndarray:
    """NED acceleration from true proportional navigation, shape (3,).

    Sightlines are directions from the drone toward the target. Closing speed
    is the velocity component along the current unit sightline. The first
    sample and any non-positive dt produce zero acceleration.
    """
    if previous_sightline_ned is None or dt <= 0.0:
        return np.zeros(3)

    velocity = np.asarray(velocity_ned, dtype=float)
    sightline = np.asarray(sightline_ned, dtype=float)
    previous = np.asarray(previous_sightline_ned, dtype=float)
    u = sightline / np.linalg.norm(sightline)
    u_prev = previous / np.linalg.norm(previous)
    closing_speed = float(np.dot(u, velocity))
    u_dot = (u - u_prev) / dt
    omega = np.cross(u, u_dot)
    return nav_constant * closing_speed * np.cross(omega, u)
