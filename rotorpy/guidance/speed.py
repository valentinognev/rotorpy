import numpy as np

COMPONENT_CAP = 19.0


def desired_velocity(direction_ned: np.ndarray, component_cap: float = COMPONENT_CAP) -> np.ndarray:
    """
    Longest velocity parallel to a NED direction with each component at most `component_cap` m/s.

    The direction is not necessarily a unit vector. A zero direction returns zeros.
    """
    direction = np.asarray(direction_ned, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.zeros(3)
    unit = direction / norm
    return unit * (float(component_cap) / float(np.max(np.abs(unit))))


def max_feasible_velocity(direction_ned: np.ndarray, max_speed_m_s: float) -> np.ndarray:
    """Longest velocity parallel to a NED direction with norm at most `max_speed_m_s`.

    The maximum-feasible-speed objective (plan Phase 7) is a total-speed (norm)
    bound, not `desired_velocity`'s per-component cap: a diagonal direction is not
    allowed to go faster just because its components each individually fit under a
    cap. The direction is not necessarily a unit vector. A zero direction returns
    zeros.
    """
    direction = np.asarray(direction_ned, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm == 0.0:
        return np.zeros(3)
    return direction * (float(max_speed_m_s) / norm)
