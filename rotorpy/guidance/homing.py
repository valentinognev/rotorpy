"""Camera homing command: lock, proportional navigation, and flat output."""

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import pixel_to_camera_ray
from rotorpy.guidance.frames import ned_to_sim, sim_body_to_frd, sim_to_ned
from rotorpy.guidance.navigation import pn_acceleration
from rotorpy.guidance.pixels import target_centroid, target_pixel_count
from rotorpy.guidance.sightline import ground_relative_position
from rotorpy.guidance.speed import desired_velocity


class HomingCommand(object):
    """One homing command. Vector fields have shape (3,)."""

    def __init__(
        self,
        velocity_ned: np.ndarray,
        acceleration_ned: np.ndarray,
        velocity_body_frd: np.ndarray,
        yaw_rad: float,
        locked: bool,
        tracking: bool,
    ) -> None:
        self.velocity_ned = np.array(velocity_ned, dtype=float, copy=True)
        self.acceleration_ned = np.array(acceleration_ned, dtype=float, copy=True)
        self.velocity_body_frd = np.array(velocity_body_frd, dtype=float, copy=True)
        self.yaw_rad = float(yaw_rad)
        self.locked = bool(locked)
        self.tracking = bool(tracking)


def body_frd_velocity(velocity_ned: np.ndarray, attitude_sim_quat: np.ndarray) -> np.ndarray:
    """Express an NED velocity in body FRD.

    The attitude quaternion rotates the simulator body into the simulator
    world. At the identity quaternion, NED (vn, ve, vd) is FRD (vn, -ve, vd).
    """
    sim_velocity = ned_to_sim(velocity_ned)
    body_velocity = Rotation.from_quat(np.asarray(attitude_sim_quat, dtype=float)).inv().apply(sim_velocity)
    return sim_body_to_frd(body_velocity)


def _sightline_ned(
    centroid: np.ndarray,
    attitude_sim_quat: np.ndarray,
    intrinsics: dict,
    extrinsics: dict,
) -> np.ndarray:
    """NED sightline from a pixel centroid.

    Camera ray, inverse extrinsics into the simulator body, body FRD, then
    the vehicle attitude applied to that simulator-body vector and
    sim_to_ned. At the identity quaternion, FRD (f, r, d) is NED (f, -r, d).
    """
    camera_ray = pixel_to_camera_ray(float(centroid[0]), float(centroid[1]), intrinsics)
    body_to_camera = Rotation.from_quat(np.asarray(extrinsics["orientation"], dtype=float))
    sim_body_ray = body_to_camera.inv().apply(camera_ray)
    frd_ray = sim_body_to_frd(sim_body_ray)
    # sim_body_to_frd is an involution, so this recovers the simulator-body
    # vector. The attitude must rotate that vector, not the FRD vector.
    sim_world_ray = Rotation.from_quat(np.asarray(attitude_sim_quat, dtype=float)).apply(
        sim_body_to_frd(frd_ray)
    )
    return sim_to_ned(sim_world_ray)


def flat_output(position_sim: np.ndarray, command: HomingCommand) -> dict:
    """Flat output that tracks the homing velocity without a position fight."""
    return {
        "x": np.array(position_sim, dtype=float, copy=True),
        "x_dot": ned_to_sim(command.velocity_ned),
        "x_ddot": ned_to_sim(command.acceleration_ned),
        "x_dddot": np.zeros(3),
        "x_ddddot": np.zeros(3),
        "yaw": command.yaw_rad,
        "yaw_dot": 0.0,
    }


class HomingGuidance(object):
    """Latch on the target, then home until the track is lost."""

    def __init__(
        self,
        nav_constant: float,
        lock_pixels: int,
        initial_velocity_ned: np.ndarray,
        intrinsics: dict,
        extrinsics: dict,
    ) -> None:
        self.nav_constant = float(nav_constant)
        self.lock_pixels = int(lock_pixels)
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self._initial_velocity_ned = np.array(initial_velocity_ned, dtype=float, copy=True)
        self._velocity_ned = np.array(self._initial_velocity_ned, dtype=float, copy=True)
        self._acceleration_ned = np.zeros(3)
        self._yaw_rad = float(np.arctan2(self._initial_velocity_ned[1], self._initial_velocity_ned[0]))
        self._locked = False
        self._tracking = False
        self._lost = False
        self._last_image_time: Optional[float] = None
        self._last_sightline: Optional[np.ndarray] = None

    def update(
        self,
        t: float,
        height_m: float,
        attitude_sim_quat: np.ndarray,
        velocity_ned: np.ndarray,
        image: Optional[np.ndarray],
    ) -> HomingCommand:
        """Advance the homing law. ``image`` is None on sim steps with no new frame."""
        if image is None:
            return self._command(attitude_sim_quat)

        previous_time = self._last_image_time
        previous_sightline = self._last_sightline
        count = target_pixel_count(image)
        if count >= self.lock_pixels:
            self._locked = True

        if self._lost:
            self._last_image_time = float(t)
            return self._command(attitude_sim_quat)

        relative = None
        sightline = None
        centroid = target_centroid(image)
        if centroid is not None:
            sightline = _sightline_ned(centroid, attitude_sim_quat, self.intrinsics, self.extrinsics)
            relative = ground_relative_position(height_m, sightline)

        if self._locked and count >= self.lock_pixels and relative is not None:
            if previous_time is None:
                dt = 0.0
            else:
                dt = float(t) - float(previous_time)
            self._acceleration_ned = pn_acceleration(
                self.nav_constant,
                velocity_ned,
                sightline,
                previous_sightline,
                dt,
            )
            self._velocity_ned = desired_velocity(relative)
            self._yaw_rad = float(np.arctan2(sightline[1], sightline[0]))
            self._tracking = True
            self._last_sightline = sightline / np.linalg.norm(sightline)
            self._last_image_time = float(t)
            return self._command(attitude_sim_quat)

        self._last_image_time = float(t)
        if self._locked:
            self._lost = True
            self._tracking = False
            self._acceleration_ned = np.zeros(3)
            return self._command(attitude_sim_quat)

        self._tracking = False
        self._velocity_ned = np.array(self._initial_velocity_ned, dtype=float, copy=True)
        self._acceleration_ned = np.zeros(3)
        return self._command(attitude_sim_quat)

    def _command(self, attitude_sim_quat: np.ndarray) -> HomingCommand:
        velocity = np.array(self._velocity_ned, dtype=float, copy=True)
        return HomingCommand(
            velocity,
            np.array(self._acceleration_ned, dtype=float, copy=True),
            body_frd_velocity(velocity, attitude_sim_quat),
            self._yaw_rad,
            self._locked,
            self._tracking,
        )
