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


def pn_lead_point(
    nav_constant: float,
    relative_position_ned: np.ndarray,
    relative_velocity_ned: np.ndarray,
    lead_time_s: float,
) -> np.ndarray:
    """Lead-pursuit aim point: `relative_position_ned`, advanced along the target's own
    estimated relative velocity by `nav_constant * lead_time_s`.

    Folds proportional navigation into the *velocity* command itself (plan Phase 7's
    "coherent single command path"), rather than only into a separate acceleration
    field a mavlink setpoint mask may ignore: steering toward where the target is
    estimated to be, not just where it was last seen. A stationary (or unobservable,
    zero-velocity) target's lead point is its relative position unchanged; the lead
    scales linearly with `nav_constant`, so changing it demonstrably changes the aim
    point (and therefore the transmitted velocity command) whenever the target's own
    estimated motion is nonzero -- the acceptance test for "PN integration" (plan
    Phase 7). `lead_time_s` is a named, proposed (not measured) design constant
    (`homing_config.PROPOSED_PN_LEAD_TIME_S`), not a physical time-to-go.
    """
    relative = np.asarray(relative_position_ned, dtype=float)
    velocity = np.asarray(relative_velocity_ned, dtype=float)
    return relative + float(nav_constant) * float(lead_time_s) * velocity
