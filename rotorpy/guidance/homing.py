"""Camera homing command: lock, proportional navigation, and flat output."""

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import pixel_to_camera_ray
from rotorpy.guidance.frames import ned_to_sim, sim_body_to_frd, sim_to_ned
from rotorpy.guidance.navigation import pn_acceleration
from rotorpy.guidance.pixels import target_pixel_count, target_square
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
        source: Optional[str],
        tracker_center: Optional[np.ndarray],
        tracker_side: Optional[float],
    ) -> None:
        self.velocity_ned = np.array(velocity_ned, dtype=float, copy=True)
        self.acceleration_ned = np.array(acceleration_ned, dtype=float, copy=True)
        self.velocity_body_frd = np.array(velocity_body_frd, dtype=float, copy=True)
        self.yaw_rad = float(yaw_rad)
        self.locked = bool(locked)
        self.tracking = bool(tracking)
        self.source = source
        if tracker_center is None:
            self.tracker_center = None
        else:
            self.tracker_center = np.array(tracker_center, dtype=float, copy=True)
        self.tracker_side = None if tracker_side is None else float(tracker_side)


def body_frd_velocity(velocity_ned: np.ndarray, attitude_sim_quat: np.ndarray) -> np.ndarray:
    """Express an NED velocity in body FRD.

    The attitude quaternion rotates the simulator body into the simulator
    world. At the identity quaternion, NED (vn, ve, vd) is FRD (vn, -ve, vd).
    """
    sim_velocity = ned_to_sim(velocity_ned)
    body_velocity = Rotation.from_quat(np.asarray(attitude_sim_quat, dtype=float)).inv().apply(sim_velocity)
    return sim_body_to_frd(body_velocity)


def _sightline_ned(
    square_center: np.ndarray,
    attitude_sim_quat: np.ndarray,
    intrinsics: dict,
    extrinsics: dict,
) -> np.ndarray:
    """NED sightline from the tracking-square center in (column, row) order.

    Camera ray, inverse extrinsics into the simulator body, body FRD, then
    the vehicle attitude applied to that simulator-body vector and
    sim_to_ned. At the identity quaternion, FRD (f, r, d) is NED (f, -r, d).
    """
    camera_ray = pixel_to_camera_ray(float(square_center[0]), float(square_center[1]), intrinsics)
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
    """Home from the forward camera until the upward camera latches, then from upward."""

    def __init__(
        self,
        nav_constant: float,
        lock_pixels: int,
        initial_velocity_ned: np.ndarray,
        intrinsics: dict,
        forward_extrinsics: dict,
        upward_extrinsics: dict,
    ) -> None:
        self.nav_constant = float(nav_constant)
        self.lock_pixels = int(lock_pixels)
        self.intrinsics = intrinsics
        self.forward_extrinsics = forward_extrinsics
        self.upward_extrinsics = upward_extrinsics
        self._initial_velocity_ned = np.array(initial_velocity_ned, dtype=float, copy=True)
        self._velocity_ned = np.array(self._initial_velocity_ned, dtype=float, copy=True)
        self._acceleration_ned = np.zeros(3)
        self._yaw_rad = float(np.arctan2(self._initial_velocity_ned[1], self._initial_velocity_ned[0]))
        self._locked = False
        self._tracking = False
        self._lost = False
        self._upward_latched = False
        self._upward_steered = False
        self._source: Optional[str] = None
        self._tracker_center: Optional[np.ndarray] = None
        self._tracker_side: Optional[float] = None
        self._last_image_time: Optional[float] = None
        self._last_sightline: Optional[np.ndarray] = None

    def update(
        self,
        t: float,
        height_m: float,
        attitude_sim_quat: np.ndarray,
        velocity_ned: np.ndarray,
        forward_image: Optional[np.ndarray],
        upward_image: Optional[np.ndarray],
    ) -> HomingCommand:
        """Advance the homing law.

        Both images None means no new frame. One None image is a camera tick
        whose missing picture has no target pixels.
        """
        if forward_image is None and upward_image is None:
            return self._command(attitude_sim_quat)
        if self._lost:
            return self._command(attitude_sim_quat)

        forward_sees = self._sees(forward_image)
        upward_sees = self._sees(upward_image)
        if forward_sees or upward_sees:
            self._locked = True
        if upward_sees:
            self._upward_latched = True

        if self._upward_latched and not upward_sees:
            return self._permanent_loss(attitude_sim_quat)

        if self._upward_latched:
            source = "upward"
            image = upward_image
            extrinsics = self.upward_extrinsics
        elif forward_sees:
            source = "forward"
            image = forward_image
            extrinsics = self.forward_extrinsics
        else:
            return self._hold(attitude_sim_quat)

        if image is None:
            return self._hold(attitude_sim_quat)
        square = target_square(image)
        if square is None:
            return self._hold(attitude_sim_quat)
        center = np.array([square[0], square[1]], dtype=float)
        sightline = _sightline_ned(center, attitude_sim_quat, self.intrinsics, extrinsics)
        relative = ground_relative_position(height_m, sightline)
        if relative is None:
            return self._hold(attitude_sim_quat)
        return self._steer(
            t, attitude_sim_quat, velocity_ned, source, center, square[2], sightline, relative,
        )

    def _sees(self, image: Optional[np.ndarray]) -> bool:
        count = 0 if image is None else target_pixel_count(image)
        return count >= self.lock_pixels

    def _hold(self, attitude_sim_quat: np.ndarray) -> HomingCommand:
        """Camera tick that is not steering: zero acceleration, and not a permanent loss."""
        self._tracking = False
        self._acceleration_ned = np.zeros(3)
        self._source = None
        self._tracker_center = None
        self._tracker_side = None
        if not self._locked:
            self._velocity_ned = np.array(self._initial_velocity_ned, dtype=float, copy=True)
        return self._command(attitude_sim_quat)

    def _permanent_loss(self, attitude_sim_quat: np.ndarray) -> HomingCommand:
        self._lost = True
        self._tracking = False
        self._acceleration_ned = np.zeros(3)
        self._source = None
        self._tracker_center = None
        self._tracker_side = None
        return self._command(attitude_sim_quat)

    def _steer(
        self,
        t: float,
        attitude_sim_quat: np.ndarray,
        velocity_ned: np.ndarray,
        source: str,
        center: np.ndarray,
        side: int,
        sightline: np.ndarray,
        relative: np.ndarray,
    ) -> HomingCommand:
        if source == "upward" and not self._upward_steered:
            previous_sightline = None
            dt = 0
        elif self._last_image_time is None:
            previous_sightline = self._last_sightline
            dt = 0.0
        else:
            previous_sightline = self._last_sightline
            dt = float(t) - float(self._last_image_time)
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
        self._source = source
        self._tracker_center = np.array(center, dtype=float, copy=True)
        self._tracker_side = float(side)
        if source == "upward":
            self._upward_steered = True
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
            self._source,
            None if self._tracker_center is None else np.array(
                self._tracker_center, dtype=float, copy=True
            ),
            self._tracker_side,
        )
