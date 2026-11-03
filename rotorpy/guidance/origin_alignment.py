"""Where the estimator's local frame sits in the simulator world.

PX4 reports its local NED position from an origin it picks itself, and a
height that is not necessarily height above the simulator ground. Nothing here
assumes the Gazebo world coordinates and the estimator coordinates are equal.
The alignment is an explicit translation, a heading offset, and a ground plane.
"""

import math

import numpy as np


def _vector3(value, name: str) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("{} must be three finite numbers".format(name))
    return vector


def _rotation_about_down(angle_rad: float) -> np.ndarray:
    cosine, sine = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


class EstimatorOriginAlignment(object):
    """Maps between the world NED frame and the estimator's local NED frame.

    `origin_world_ned` is the estimator origin written in the world frame.
    `yaw_offset_rad` is the heading, clockwise from world north, of the
    estimator's north axis. `ground_down_world_m` is the world NED down
    coordinate of the ground plane (zero for flat ground at the world origin).
    """

    def __init__(self, origin_world_ned, yaw_offset_rad: float, ground_down_world_m: float) -> None:
        self.origin_world_ned = _vector3(origin_world_ned, "origin_world_ned")
        self.yaw_offset_rad = float(yaw_offset_rad)
        self.ground_down_world_m = float(ground_down_world_m)
        if not (math.isfinite(self.yaw_offset_rad) and math.isfinite(self.ground_down_world_m)):
            raise ValueError("yaw offset and ground height must be finite")
        self._world_from_estimator = _rotation_about_down(self.yaw_offset_rad)

    def estimator_to_world(self, position_ned) -> np.ndarray:
        """A position in the estimator frame, in the world frame."""
        return self.origin_world_ned + self._world_from_estimator @ _vector3(position_ned, "position")

    def world_to_estimator(self, position_ned) -> np.ndarray:
        """A world position, in the estimator frame."""
        return self._world_from_estimator.T @ (_vector3(position_ned, "position") - self.origin_world_ned)

    def estimator_vector_to_world(self, vector_ned) -> np.ndarray:
        """A velocity or direction in the estimator frame, in the world frame. No translation."""
        return self._world_from_estimator @ _vector3(vector_ned, "vector")

    def world_vector_to_estimator(self, vector_ned) -> np.ndarray:
        """A world velocity or direction, in the estimator frame. No translation."""
        return self._world_from_estimator.T @ _vector3(vector_ned, "vector")

    def heading_to_world(self, heading_rad: float) -> float:
        """An estimator heading as a world heading."""
        return float(heading_rad) + self.yaw_offset_rad

    def heading_to_estimator(self, heading_rad: float) -> float:
        """A world heading as an estimator heading."""
        return float(heading_rad) - self.yaw_offset_rad

    @property
    def ground_down_in_estimator_m(self) -> float:
        """The estimator NED down coordinate of the ground plane."""
        return self.ground_down_world_m - float(self.origin_world_ned[2])

    def height_above_ground(self, estimator_position_ned) -> float:
        """Height above the world ground plane of a position in the estimator frame."""
        down = float(_vector3(estimator_position_ned, "position")[2])
        return self.ground_down_in_estimator_m - down
