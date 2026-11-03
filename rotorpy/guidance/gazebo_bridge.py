"""Gazebo ENU state into the frames the homing law already uses."""

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.frames import enu_to_ned

# (east, north, up) -> (north, east, up). Det -1, paired with the body flip below.
_ENU_TO_SIM = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
)
# Gazebo body +Y is left. The homing body +Y points east at the identity yaw.
_BODY_Y_FLIP = np.diag([1.0, -1.0, 1.0])


def attitude_gazebo_to_homing(q_xyzw: np.ndarray) -> np.ndarray:
    """Scalar-last Gazebo FLU-in-ENU quaternion to the homing simulator quaternion."""
    gazebo = Rotation.from_quat(np.array(q_xyzw, dtype=float, copy=True))
    mapped = _ENU_TO_SIM @ gazebo.as_matrix() @ _BODY_Y_FLIP
    return Rotation.from_matrix(mapped).as_quat()


def gazebo_position_to_sim(position_enu: np.ndarray) -> np.ndarray:
    """Gazebo position (east, north, up) to the homing simulator (north, east, up)."""
    east, north, up = np.asarray(position_enu, dtype=float)
    return np.array([north, east, up], dtype=float)


def velocity_gazebo_to_ned(velocity_enu: np.ndarray) -> np.ndarray:
    """Gazebo world velocity (east, north, up) to NED."""
    return enu_to_ned(np.asarray(velocity_enu, dtype=float))
