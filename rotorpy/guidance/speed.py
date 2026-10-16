import numpy as np

COMPONENT_CAP = 19.0


def desired_velocity(direction_ned: np.ndarray) -> np.ndarray:
    """
    Longest velocity parallel to a NED direction with each component at most COMPONENT_CAP m/s.

    The direction is not necessarily a unit vector. A zero direction returns zeros.
    """
    direction = np.asarray(direction_ned, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.zeros(3)
    unit = direction / norm
    return unit * (COMPONENT_CAP / float(np.max(np.abs(unit))))
