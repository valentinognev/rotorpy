"""Conversions between the FLU-ENU and FRD-NED frame pairs.

World vectors map (east, north, up) to (north, east, -up). Body vectors map
(forward, left, up) to (forward, -left, -up). Each vector conversion copies
its input. Quaternions stay scalar-last [x, y, z, w].
"""

import math
from enum import Enum
from typing import Any, Dict, List, Sequence

import numpy as np
from scipy.spatial.transform import Rotation


class FrameConvention(str, Enum):
    """Which world/body pair a simulation is using."""

    FLU_ENU = "flu_enu"
    FRD_NED = "frd_ned"


M_ENU_TO_NED = np.array(
    [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
)
M_FLU_TO_FRD = np.diag([1.0, -1.0, -1.0])

_FLAT_VECTOR_KEYS = ("x", "x_dot", "x_ddot", "x_dddot", "x_ddddot")


def _mapped_vector(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    copied = np.array(vector, dtype=float, copy=True)
    return matrix @ copied


def enu_to_ned(vector: np.ndarray) -> np.ndarray:
    """Convert an ENU vector of shape (3,) to NED."""
    return _mapped_vector(M_ENU_TO_NED, vector)


def ned_to_enu(vector: np.ndarray) -> np.ndarray:
    """Convert a NED vector of shape (3,) to ENU."""
    return _mapped_vector(M_ENU_TO_NED, vector)


def flu_to_frd(vector: np.ndarray) -> np.ndarray:
    """Convert a FLU body vector of shape (3,) to FRD."""
    return _mapped_vector(M_FLU_TO_FRD, vector)


def frd_to_flu(vector: np.ndarray) -> np.ndarray:
    """Convert an FRD body vector of shape (3,) to FLU."""
    return _mapped_vector(M_FLU_TO_FRD, vector)


def _attitude_between_pairs(q_xyzw: np.ndarray) -> np.ndarray:
    rotation = Rotation.from_quat(np.array(q_xyzw, dtype=float, copy=True))
    mapped = M_ENU_TO_NED @ rotation.as_matrix() @ M_FLU_TO_FRD
    return Rotation.from_matrix(mapped).as_quat()


def attitude_enu_flu_to_ned_frd(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert a scalar-last ENU/FLU attitude to NED/FRD."""
    return _attitude_between_pairs(q_xyzw)


def attitude_ned_frd_to_enu_flu(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert a scalar-last NED/FRD attitude to ENU/FLU.

    ``M_ENU_TO_NED`` and ``M_FLU_TO_FRD`` each square to identity, so the
    inverse is the same sandwich as the forward map.
    """
    return _attitude_between_pairs(q_xyzw)


def yaw_enu_to_ned(yaw: float) -> float:
    """Convert a yaw about world up from ENU to NED."""
    return math.pi / 2.0 - float(yaw)


def yaw_ned_to_enu(yaw: float) -> float:
    """Convert a yaw about world up from NED to ENU.

    ``yaw_ned = pi/2 - yaw_enu``, so the inverse uses the same expression.
    """
    return math.pi / 2.0 - float(yaw)


def yaw_rate_enu_to_ned(yaw_dot: float) -> float:
    """Convert a yaw rate from ENU to NED."""
    return -float(yaw_dot)


def aabb_enu_to_ned(extents: Sequence[float]) -> List[float]:
    """Swap the horizontal intervals and negate the vertical interval."""
    xmin, xmax, ymin, ymax, zmin, zmax = extents
    return [
        float(ymin),
        float(ymax),
        float(xmin),
        float(xmax),
        float(-zmax),
        float(-zmin),
    ]


def flat_enu_to_ned(flat: Dict[str, Any]) -> Dict[str, Any]:
    """Convert flat-output position derivatives and yaw into NED."""
    converted = dict(flat)
    for key in _FLAT_VECTOR_KEYS:
        if key in converted:
            converted[key] = enu_to_ned(converted[key])
    if "yaw" in converted:
        converted["yaw"] = yaw_enu_to_ned(converted["yaw"])
    if "yaw_dot" in converted:
        converted["yaw_dot"] = yaw_rate_enu_to_ned(converted["yaw_dot"])
    return converted


def body_to_sensor_flu_to_frd(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert a body-to-sensor quaternion from FLU to FRD.

    The active rotation is ``R_new = R_old @ M_FLU_TO_FRD``.
    """
    rotation = Rotation.from_quat(np.array(q_xyzw, dtype=float, copy=True))
    mapped = rotation.as_matrix() @ M_FLU_TO_FRD
    return Rotation.from_matrix(mapped).as_quat()


def apply_frame(frame: FrameConvention, *objects: Any) -> None:
    """Call ``set_frame`` on each object that defines it."""
    for obj in objects:
        if obj is None:
            continue
        setter = getattr(obj, "set_frame", None)
        if setter is None:
            continue
        setter(frame)
