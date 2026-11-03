"""Camera homing command: lock, proportional navigation, and flat output."""

from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import CameraMount, pixel_ray_ned, pixel_to_camera_ray
from rotorpy.guidance.frames import ned_to_sim, sim_body_to_frd, sim_to_ned
from rotorpy.guidance.gazebo_bridge import attitude_homing_to_ned_frd, heading_from_homing_attitude
from rotorpy.guidance.navigation import pn_acceleration
from rotorpy.guidance.pixels import target_pixel_count, target_square
from rotorpy.guidance.sightline import ground_relative_position, reference_relative_position
from rotorpy.guidance.speed import COMPONENT_CAP, desired_velocity

# Default below this horizontal speed the measured velocity has no usable heading, when
# the caller does not pass `min_course_speed_m_s`. The Gazebo config path overrides this
# with the named `PROPOSED_MIN_COURSE_SPEED_M_S` in `homing_config.py`.
_MIN_COURSE_SPEED_M_S = 1.0


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


_GRAVITY = 9.81
_TRACK_TAU = 0.5
_MAX_TILT_RAD = np.deg2rad(45.0)
_MIN_SPECIFIC_UP = 0.3 * _GRAVITY


def track_acceleration(
    velocity_sim: np.ndarray,
    goal_sim: np.ndarray,
    pn_sim: np.ndarray,
    tau_s: float = _TRACK_TAU,
    max_tilt_deg: float = 45.0,
    min_specific_up_g: float = 0.3,
) -> np.ndarray:
    """Acceleration toward `goal_sim` that keeps the specific force upright.

    The request is proportional navigation plus the velocity miss over half a
    second. It is scaled, never reshaped, until the specific force stays
    upright. Specific force is that acceleration plus gravity on simulator +z.
    Its up component stays at least 0.3 g, and its tilt from +z stays within
    45°. A raw velocity step through the SE3 law asks for a downward force,
    and the attitude on that force is the long arc.
    """
    velocity = np.asarray(velocity_sim, dtype=float)
    goal = np.asarray(goal_sim, dtype=float)
    pn = np.asarray(pn_sim, dtype=float)
    requested = np.array(pn + (goal - velocity) / float(tau_s), dtype=float, copy=True)
    return _largest_upright_scale(
        requested,
        max_tilt_rad=np.deg2rad(max_tilt_deg),
        min_specific_up=float(min_specific_up_g) * _GRAVITY,
    ) * requested


def _largest_upright_scale(
    requested: np.ndarray,
    max_tilt_rad: float = _MAX_TILT_RAD,
    min_specific_up: float = _MIN_SPECIFIC_UP,
) -> float:
    """Largest scale in [0, 1] whose specific force stays inside the upright cone.

    Scaling the whole request keeps it parallel to the velocity miss. Clamping
    one axis and then another turns a pursuit correction into a dive.
    """
    tan_tilt = float(np.tan(max_tilt_rad))
    horizontal = float(np.linalg.norm(requested[:2]))
    vertical = float(requested[2])
    scale = 1.0
    if vertical < 0.0:
        scale = min(scale, (_GRAVITY - min_specific_up) / (-vertical))
    # k * H <= (k * vertical + g) * tan(tilt)
    denom = horizontal - vertical * tan_tilt
    if denom > 0.0:
        scale = min(scale, (_GRAVITY * tan_tilt) / denom)
    return max(0.0, scale)


def flat_output(
    position_sim: np.ndarray,
    command: HomingCommand,
    velocity_sim: np.ndarray,
    tau_s: float = _TRACK_TAU,
    max_tilt_deg: float = 45.0,
    min_specific_up_g: float = 0.3,
) -> dict:
    """Flat output that tracks the homing velocity without a position fight.

    The velocity setpoint is the measured velocity, so the SE3 velocity term
    adds nothing. `track_acceleration` carries the homing request.
    """
    measured = np.array(velocity_sim, dtype=float, copy=True)
    return {
        "x": np.array(position_sim, dtype=float, copy=True),
        "x_dot": measured,
        "x_ddot": track_acceleration(
            measured,
            ned_to_sim(command.velocity_ned),
            ned_to_sim(command.acceleration_ned),
            tau_s=tau_s,
            max_tilt_deg=max_tilt_deg,
            min_specific_up_g=min_specific_up_g,
        ),
        "x_dddot": np.zeros(3),
        "x_ddddot": np.zeros(3),
        "yaw": command.yaw_rad,
        "yaw_dot": 0.0,
    }


class HomingGuidance(object):
    """Home from the forward camera until the upward camera latches, then from upward.

    Two camera geometries exist. Without mounts, the legacy RotorPy example
    geometry converts pixels through `forward_extrinsics` into its simulator
    frames, where the body's +y points east at the identity yaw. With
    `forward_mount` and `upward_mount`, pixels become NED rays through the
    true camera frame, body FRD and the vehicle attitude, including each
    camera's translation. In both, `attitude_sim_quat` is the homing
    simulator quaternion that `attitude_gazebo_to_homing` produces.

    With no `initial_velocity_ned`, the course held before lock is
    `initial_speed` along the vehicle's own heading at its first update. The
    guidance never sees where the target is.
    """

    def __init__(
        self,
        nav_constant: float,
        lock_pixels: int,
        initial_velocity_ned: Optional[np.ndarray],
        intrinsics: dict,
        forward_extrinsics: Optional[dict],
        upward_extrinsics: Optional[dict],
        component_cap: float = COMPONENT_CAP,
        initial_speed: Optional[float] = None,
        forward_mount: Optional[CameraMount] = None,
        upward_mount: Optional[CameraMount] = None,
        target_reference_height_m: float = 0.0,
        min_course_speed_m_s: float = _MIN_COURSE_SPEED_M_S,
    ) -> None:
        if initial_velocity_ned is None and initial_speed is None:
            raise ValueError("give an initial velocity or an initial speed")
        if (forward_mount is None) != (upward_mount is None):
            raise ValueError("give both camera mounts or neither")
        if forward_mount is None and (forward_extrinsics is None or upward_extrinsics is None):
            raise ValueError("legacy geometry needs both extrinsics")
        self.nav_constant = float(nav_constant)
        self.lock_pixels = int(lock_pixels)
        self._component_cap = float(component_cap)
        self.intrinsics = intrinsics
        self.forward_extrinsics = forward_extrinsics
        self.upward_extrinsics = upward_extrinsics
        self._mounts = None if forward_mount is None else {"forward": forward_mount, "upward": upward_mount}
        self._reference_height_m = float(target_reference_height_m)
        self._min_course_speed_m_s = float(min_course_speed_m_s)
        self._initial_speed = None if initial_speed is None else float(initial_speed)
        self._acceleration_ned = np.zeros(3)
        self._initial_velocity_ned: Optional[np.ndarray] = None
        self._velocity_ned = np.zeros(3)
        self._yaw_rad = 0.0
        if initial_velocity_ned is not None:
            self._set_course(np.array(initial_velocity_ned, dtype=float, copy=True))
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

    def _set_course(self, velocity_ned: np.ndarray) -> None:
        self._initial_velocity_ned = velocity_ned
        self._velocity_ned = np.array(velocity_ned, dtype=float, copy=True)
        self._yaw_rad = float(np.arctan2(velocity_ned[1], velocity_ned[0]))

    def _adopt_course(self, velocity_ned: np.ndarray, attitude_sim_quat: np.ndarray) -> None:
        """Hold the vehicle's own heading at the configured speed until the target is seen."""
        velocity = np.asarray(velocity_ned, dtype=float)
        if float(np.hypot(velocity[0], velocity[1])) >= self._min_course_speed_m_s:
            heading = float(np.arctan2(velocity[1], velocity[0]))
        else:
            heading = heading_from_homing_attitude(attitude_sim_quat)
        speed = self._initial_speed
        self._set_course(np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0]))

    def _sightline_and_relative(
        self,
        source: str,
        center: np.ndarray,
        height_m: float,
        attitude_sim_quat: np.ndarray,
    ):
        """Unit sightline and body-to-target vector, or (None, None) without a usable bearing.

        A reference-plane hit (`reference_relative_position`) is unreachable once height
        drops to or below `target_reference_height_m` (final-approach altitudes), not only
        when the ray is aimed above the horizon. Freezing the command in the former case
        would hold a stale velocity indefinitely as the vehicle keeps descending, so that
        case falls back to steering from the camera-ray bearing alone (the existing speed
        law only uses direction, so the bearing is a correct, if offset-free, substitute).
        The latter case (the ray truly never reaches any downward plane) has no bearing
        either and stays a hold, same as before.
        """
        if self._mounts is None:
            extrinsics = self.forward_extrinsics if source == "forward" else self.upward_extrinsics
            sightline = _sightline_ned(center, attitude_sim_quat, self.intrinsics, extrinsics)
            return sightline, ground_relative_position(height_m, sightline)
        rotation = Rotation.from_quat(attitude_homing_to_ned_frd(attitude_sim_quat)).as_matrix()
        offset, direction = pixel_ray_ned(
            float(center[0]), float(center[1]), self.intrinsics, self._mounts[source], rotation
        )
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0 or direction[2] <= 0.0:
            return None, None
        relative = reference_relative_position(height_m, offset, direction, self._reference_height_m)
        if relative is None:
            # Fall back to the pure camera-ray bearing (no camera-translation correction):
            # an approximation, but keeps steering instead of freezing the command.
            bearing = direction / norm
            return bearing, bearing
        return relative / np.linalg.norm(relative), relative

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
        if self._initial_velocity_ned is None:
            self._adopt_course(velocity_ned, attitude_sim_quat)
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
        elif forward_sees:
            source = "forward"
            image = forward_image
        else:
            return self._hold(attitude_sim_quat)

        if image is None:
            return self._hold(attitude_sim_quat)
        square = target_square(image)
        if square is None:
            return self._hold(attitude_sim_quat)
        center = np.array([square[0], square[1]], dtype=float)
        sightline, relative = self._sightline_and_relative(source, center, height_m, attitude_sim_quat)
        if relative is None:
            return self._hold(attitude_sim_quat)
        return self._steer(
            t, attitude_sim_quat, velocity_ned, source, center, square[2], sightline, relative,
        )

    def reset_derivative_history(self) -> None:
        """Clear the proportional-navigation derivative history (previous sightline/time).

        Call this after a `VehicleState` clock discontinuity (sim-time reset, backward
        jump, or detected pause) is reported: the previous sightline/timestamp pair was
        recorded before the discontinuity, and feeding it into the next line-of-sight-rate
        computation would fabricate a spurious angular rate from two readings that are no
        longer actually adjacent in (sim) time.
        """
        self._last_sightline = None
        self._last_image_time = None

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
        self._velocity_ned = desired_velocity(relative, self._component_cap)
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
        if self._mounts is None:
            body_velocity = body_frd_velocity(velocity, attitude_sim_quat)
        else:
            rotation = Rotation.from_quat(attitude_homing_to_ned_frd(attitude_sim_quat))
            body_velocity = rotation.inv().apply(velocity)
        return HomingCommand(
            velocity,
            np.array(self._acceleration_ned, dtype=float, copy=True),
            body_velocity,
            self._yaw_rad,
            self._locked,
            self._tracking,
            self._source,
            None if self._tracker_center is None else np.array(
                self._tracker_center, dtype=float, copy=True
            ),
            self._tracker_side,
        )
