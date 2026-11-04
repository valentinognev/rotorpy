"""Camera homing command: lock, proportional navigation, and flat output."""

from typing import Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import CameraMount, pixel_ray_ned, pixel_to_camera_ray
from rotorpy.guidance.command_limits import VehicleEnvelope, limit_command
from rotorpy.guidance.frames import ned_to_sim, sim_body_to_frd, sim_to_ned
from rotorpy.guidance.gazebo_bridge import attitude_homing_to_ned_frd, heading_from_homing_attitude
from rotorpy.guidance.navigation import pn_acceleration, pn_lead_point
from rotorpy.guidance.pixels import (
    DEFAULT_CONFIG,
    DetectionStatus,
    DetectorConfig,
    TargetAssociator,
    TargetObservation,
    target_pixel_count,
    target_square,
)
from rotorpy.guidance.sightline import ground_relative_position, reference_relative_position
from rotorpy.guidance.speed import COMPONENT_CAP, desired_velocity, max_feasible_velocity
from rotorpy.guidance.target_estimator import TargetMotionEstimate, TargetMotionEstimator, TargetMotionStatus
from rotorpy.guidance.tracking import CameraSignal, TrackingState, TrackingStateMachine

# Default below this horizontal speed the measured velocity has no usable heading, when
# the caller does not pass `min_course_speed_m_s`. The Gazebo config path overrides this
# with the named `PROPOSED_MIN_COURSE_SPEED_M_S` in `homing_config.py`.
_MIN_COURSE_SPEED_M_S = 1.0

# Private literal defaults matching `homing_config`'s named `PROPOSED_*` tracking
# thresholds (see `tracking_thresholds()` there): kept here, not imported from
# `homing_config`, so direct construction (every existing test) keeps working without
# naming every threshold, and so `homing_config` (which imports `HomingGuidance`) never
# has to be imported back by this module.
_ACQUIRE_FRAMES = 1
_REACQUIRE_FRAMES = 1
_CAMERA_SWITCH_MARGIN = 0.1
_MAX_PREDICTION_GAP_S = 2.0
_MAX_TRACK_UNCERTAINTY_PX = 150.0
# Private literal defaults matching `homing_config`'s named `PROPOSED_*` target-motion
# thresholds (see `target_motion_thresholds()` there); same reasoning as the tracking
# thresholds above.
_TARGET_MAX_SPEED_M_S = 5.0
_TARGET_MAX_ACCEL_M_S2 = 2.0
_TARGET_OUTLIER_CHI2_THRESHOLD = 9.21
_TARGET_MAX_AGE_S = 1.0
_TARGET_MAX_POSITION_SIGMA_M = 5.0
_TARGET_CLIPPED_INFLATION = 3.0
_TARGET_LOW_CONFIDENCE_INFLATION = 2.0
_TARGET_CONFIDENCE_FLOOR = 0.2
# Private literal defaults matching `homing_config`'s named `PROPOSED_*` vehicle-
# envelope/PN-lead/feedforward thresholds (plan Phase 7; see `vehicle_envelope()`
# there): same reasoning as the tracking/target-motion threshold groups above.
_VEHICLE_MAX_SPEED_M_S = 15.0
_VEHICLE_MAX_ACCEL_M_S2 = 5.0
_VEHICLE_MAX_JERK_M_S3 = 10.0
_VEHICLE_MAX_YAW_RATE_RAD_S = 1.0
_VEHICLE_MAX_VERTICAL_SPEED_M_S = 8.0
_VISIBILITY_MARGIN_FRAC = 0.2
_VISIBILITY_MIN_SPEED_SCALE = 0.3
_PN_LEAD_TIME_S = 0.2
_ENABLE_ACCELERATION_FEEDFORWARD = False
# Documented design (unchanged by this module): home from the forward camera until the
# upward camera latches, then home from upward -- so upward outranks forward whenever
# both are visible, not just the first time upward is ever seen.
_CAMERA_PRIORITY = ("upward", "forward")


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
        detections: Optional[Dict[str, TargetObservation]] = None,
        tracking_state: Optional[TrackingState] = None,
        abort_reason: Optional[str] = None,
        target_motion: Optional[TargetMotionEstimate] = None,
        requested_velocity_ned: Optional[np.ndarray] = None,
        limiting_reasons: Optional[List[str]] = None,
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
        # Per-camera `pixels.TargetObservation` for this tick, Gazebo (mounts) path only.
        # `None` on the legacy no-mounts path, which keeps target_square/target_pixel_count.
        self.detections = detections
        # `tracking.TrackingState` this tick ended in. `ABORT` is the guidance's own
        # explicit, one-way failure response to a prolonged loss or runaway prediction
        # uncertainty: the documented, configurable decision is that a caller sees this
        # and maps it to the T1 `RunOutcome.TARGET_LOST` outcome and the T3
        # `SetpointStream.mark_fatal()` failure-response path, instead of continuing to
        # accept this (or any later) command as a normal stale-command hold. `COMPLETE`
        # is set only by a caller's own `HomingGuidance.complete()` call (the truth-side
        # scorer's signal; guidance itself never reads truth), never internally.
        self.tracking_state = tracking_state
        # Why `tracking_state` is `ABORT`, or `None` otherwise.
        self.abort_reason = abort_reason
        # `target_estimator.TargetMotionEstimate` predicted to this tick's control
        # time (Gazebo/mounts path only; `None` on the legacy no-mounts path, which
        # does not build an estimator). Purely additive and purely advisory: no
        # existing field above (velocity_ned/acceleration_ned/yaw_rad/...) is computed
        # from this, so an uncertain or unobservable estimate never silently degrades
        # steering -- Task 7 owns actually consuming it in the speed/PN law, behind
        # its own documented fallback to the existing bearing-only/pure-pursuit
        # behavior whenever `target_motion.status` is not `OBSERVABLE`.
        self.target_motion = target_motion
        # Gazebo (mounts) path only (plan Phase 7): the velocity guidance asked for
        # before `command_limits.limit_command` bounded it to the vehicle envelope/
        # continuity/visibility limits -- `velocity_ned` above is that limited (and,
        # barring a tracker-level startup override, transmitted) value. `None` on the
        # legacy no-mounts path, which has no envelope to limit against.
        self.requested_velocity_ned = None if requested_velocity_ned is None else np.array(
            requested_velocity_ned, dtype=float, copy=True
        )
        # Every named bound `limit_command` actually applied this tick, plus
        # `"target_motion_unavailable:<reason>"` whenever steering fell back to the
        # bearing-only aim point because `target_motion` was not `OBSERVABLE`. Always
        # `[]` (never `None`) when passed at all; `requested_velocity_ned is None` is
        # the sentinel callers (`telemetry.setpoint_fields_from_command`) check for
        # "guidance had no envelope to limit against this tick" (legacy no-mounts
        # path, or a mounts-path hold/terminal tick), not this field.
        self.limiting_reasons = None if limiting_reasons is None else list(limiting_reasons)


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
        detector_config: DetectorConfig = DEFAULT_CONFIG,
        acquire_frames: int = _ACQUIRE_FRAMES,
        reacquire_frames: int = _REACQUIRE_FRAMES,
        camera_switch_margin: float = _CAMERA_SWITCH_MARGIN,
        max_prediction_gap_s: float = _MAX_PREDICTION_GAP_S,
        max_track_uncertainty_px: float = _MAX_TRACK_UNCERTAINTY_PX,
        max_speed_m_s: float = _TARGET_MAX_SPEED_M_S,
        max_accel_m_s2: float = _TARGET_MAX_ACCEL_M_S2,
        outlier_chi2_threshold: float = _TARGET_OUTLIER_CHI2_THRESHOLD,
        max_age_s: float = _TARGET_MAX_AGE_S,
        max_position_sigma_m: float = _TARGET_MAX_POSITION_SIGMA_M,
        clipped_inflation: float = _TARGET_CLIPPED_INFLATION,
        low_confidence_inflation: float = _TARGET_LOW_CONFIDENCE_INFLATION,
        confidence_floor: float = _TARGET_CONFIDENCE_FLOOR,
        vehicle_envelope: Optional[VehicleEnvelope] = None,
        pn_lead_time_s: float = _PN_LEAD_TIME_S,
        enable_acceleration_feedforward: bool = _ENABLE_ACCELERATION_FEEDFORWARD,
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
        # Structured detection (pixels.detect_target/TargetAssociator) replaces
        # target_square/target_pixel_count only on the Gazebo (mounts) path; the legacy
        # no-mounts path keeps those functions and never builds `detections`.
        self._associators = None if self._mounts is None else {
            "forward": TargetAssociator(detector_config), "upward": TargetAssociator(detector_config)
        }
        # Timestamped constant-velocity target-motion estimator (plan Phase 6), Gazebo
        # (mounts) path only: it needs the true NED ray geometry (`pixel_ray_ned`) the
        # legacy no-mounts path does not have. Task 6 wired it in purely additively;
        # Task 7's `_steer` now also consults it (`predict_to` at the control time) for
        # a lead-pursuit aim point when `status is OBSERVABLE`, falling back to the
        # existing bearing-only aim point otherwise (see `_steer`). Parameter names
        # mirror `TargetMotionEstimator.__init__` directly (same convention as the
        # tracking thresholds above), so `homing_config.target_motion_thresholds()`
        # unpacks straight into this constructor.
        self._estimator: Optional[TargetMotionEstimator] = None if self._mounts is None else TargetMotionEstimator(
            max_speed_m_s=max_speed_m_s,
            max_accel_m_s2=max_accel_m_s2,
            outlier_chi2_threshold=outlier_chi2_threshold,
            max_age_s=max_age_s,
            max_position_sigma_m=max_position_sigma_m,
            clipped_inflation=clipped_inflation,
            low_confidence_inflation=low_confidence_inflation,
            confidence_floor=confidence_floor,
        )
        # Vehicle-envelope/continuity/visibility command limiting (plan Phase 7),
        # Gazebo (mounts) path only: the legacy no-mounts path keeps its unlimited
        # `desired_velocity`/`pn_acceleration` output unchanged (every existing legacy
        # test asserts that output exactly). `self.enable_acceleration_feedforward` is
        # public and read-only from here on: the tracker shell decides whether to
        # actually send `HomingCommand.acceleration_ned` to PX4 from it (default
        # False -- masked, unchanged from before this task; see `homing_config`'s
        # `PROPOSED_ENABLE_ACCELERATION_FEEDFORWARD`). A caller passing no explicit
        # `vehicle_envelope` gets this module's own private literal defaults (same
        # convention as every other threshold group); `homing_config.guidance_from_config`
        # passes `VehicleEnvelope(**vehicle_envelope())` instead.
        self._envelope: Optional[VehicleEnvelope] = None
        if self._mounts is not None:
            self._envelope = vehicle_envelope if vehicle_envelope is not None else VehicleEnvelope(
                max_speed_m_s=_VEHICLE_MAX_SPEED_M_S,
                max_accel_m_s2=_VEHICLE_MAX_ACCEL_M_S2,
                max_jerk_m_s3=_VEHICLE_MAX_JERK_M_S3,
                max_yaw_rate_rad_s=_VEHICLE_MAX_YAW_RATE_RAD_S,
                max_vertical_speed_m_s=_VEHICLE_MAX_VERTICAL_SPEED_M_S,
                visibility_margin_frac=_VISIBILITY_MARGIN_FRAC,
                visibility_min_speed_scale=_VISIBILITY_MIN_SPEED_SCALE,
            )
        self._pn_lead_time_s = float(pn_lead_time_s)
        self.enable_acceleration_feedforward = bool(enable_acceleration_feedforward)
        # Set only by `_steer`'s mounts branch; `None`/`[]` otherwise (legacy path, or
        # a hold/terminal tick), matching `HomingCommand`'s own None-means-unset
        # convention for `requested_velocity_ned`/`limiting_reasons`.
        self._requested_velocity_ned: Optional[np.ndarray] = None
        self._limiting_reasons: List[str] = []
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
        # Explicit STARTUP -> ACQUIRE -> TRACK -> TEMPORARY_LOSS -> REACQUIRE state
        # machine, with terminal ABORT/COMPLETE; replaces the old one-way
        # "_upward_latched"/"_lost" pair, which had no recovery path (see
        # `tests/test_homing_recorded_handoff_replay.py`). Shared by both the legacy
        # (no-mounts) and Gazebo (mounts) paths: the bug it fixes was in this shared
        # camera-selection logic, not in the Gazebo-only detection pieces.
        self._tracker = TrackingStateMachine(
            acquire_frames=acquire_frames,
            reacquire_frames=reacquire_frames,
            switch_margin=camera_switch_margin,
            max_prediction_gap_s=max_prediction_gap_s,
            max_uncertainty_px=max_track_uncertainty_px,
            camera_priority=_CAMERA_PRIORITY,
        )
        self._source: Optional[str] = None
        self._tracker_center: Optional[np.ndarray] = None
        self._tracker_side: Optional[float] = None
        self._last_image_time: Optional[float] = None
        self._last_sightline: Optional[np.ndarray] = None
        self._detections: Optional[Dict[str, TargetObservation]] = None
        # This tick's control time/vehicle velocity, recorded on every `update()` call
        # (even when it returns early with no new frame): `_command` uses them to
        # `predict_to` the target-motion estimate to the current tick regardless of
        # which branch built the command.
        self._last_control_time: Optional[float] = None
        self._last_velocity_ned: Optional[np.ndarray] = None

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
        """Unit sightline, body-to-target vector, and the NED ray (origin, direction)
        behind that vector -- or (None, None, None, None) without a usable bearing.

        The ray is only returned alongside a genuine reference-plane hit (mounts path):
        it is what `_estimator.update` needs to turn pixel uncertainty into a ground-
        plane position covariance, and a bearing-only fallback or the legacy no-mounts
        path has no such plane-relative geometry to filter.

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
            return sightline, ground_relative_position(height_m, sightline), None, None
        rotation = Rotation.from_quat(attitude_homing_to_ned_frd(attitude_sim_quat)).as_matrix()
        offset, direction = pixel_ray_ned(
            float(center[0]), float(center[1]), self.intrinsics, self._mounts[source], rotation
        )
        direction = np.asarray(direction, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0 or direction[2] <= 0.0:
            return None, None, None, None
        relative = reference_relative_position(height_m, offset, direction, self._reference_height_m)
        if relative is None:
            # Fall back to the pure camera-ray bearing (no camera-translation correction):
            # an approximation, but keeps steering instead of freezing the command. No
            # usable ground-plane geometry for the estimator either, so no ray.
            bearing = direction / norm
            return bearing, bearing, None, None
        return relative / np.linalg.norm(relative), relative, offset, direction / norm

    def update(
        self,
        t: float,
        height_m: float,
        attitude_sim_quat: np.ndarray,
        velocity_ned: np.ndarray,
        forward_image: Optional[np.ndarray],
        upward_image: Optional[np.ndarray],
        control_t: Optional[float] = None,
    ) -> HomingCommand:
        """Advance the homing law.

        Both images None means no new frame. One None image is a camera tick
        whose missing picture has no target pixels.

        `t` is the measurement time: every detection/dt/estimator-update computation
        below uses it, and so must `height_m`/`attitude_sim_quat`/`velocity_ned` --
        the vehicle state *at* `t` (the Gazebo tracker samples these from the T3
        `VehicleState` history at the image's own capture time, not whatever PX4
        happened to report most recently). `control_t` is the separate, later time
        the resulting command actually takes effect (the processing instant); it
        defaults to `t` (every pre-Phase-7 caller's single-timestamp convention,
        including every direct `HomingGuidance` construction in this file's own
        tests, is unchanged). Only `_estimator.predict_to` (the delay-compensated
        target-motion estimate `_steer` aims at, and the one `_command` attaches to
        `HomingCommand.target_motion`) uses `control_t`: the processing delay between
        a frame's capture and the command it produces is exactly what `predict_to`
        exists to compensate for.
        """
        control_t = float(t) if control_t is None else float(control_t)
        self._last_control_time = control_t
        self._last_velocity_ned = np.array(velocity_ned, dtype=float, copy=True)
        if self._initial_velocity_ned is None:
            self._adopt_course(velocity_ned, attitude_sim_quat)
        if forward_image is None and upward_image is None:
            return self._command(attitude_sim_quat)
        if self._tracker.is_terminal:
            return self._terminal_command(attitude_sim_quat)

        forward_obs = self._detect("forward", forward_image, t)
        upward_obs = self._detect("upward", upward_image, t)
        if self._associators is not None:
            self._detections = {"forward": forward_obs, "upward": upward_obs}

        forward_sees = self._sees(forward_image, forward_obs)
        upward_sees = self._sees(upward_image, upward_obs)
        if forward_sees or upward_sees:
            self._locked = True

        self._tracker.update(t, {
            "forward": self._camera_signal("forward", forward_sees, forward_obs),
            "upward": self._camera_signal("upward", upward_sees, upward_obs),
        })
        if self._tracker.is_terminal:
            return self._terminal_command(attitude_sim_quat)
        if self._tracker.state is not TrackingState.TRACK:
            # ACQUIRE (not yet confirmed), TEMPORARY_LOSS or REACQUIRE: not steering
            # this tick, but not a failure either -- bounded, recoverable hold.
            return self._hold(attitude_sim_quat)

        source = self._tracker.camera
        image = forward_image if source == "forward" else upward_image
        obs = forward_obs if source == "forward" else upward_obs
        center, side = self._tracked_square(image, obs)
        if center is None:
            return self._hold(attitude_sim_quat)
        sightline, relative, ray_origin, ray_direction = self._sightline_and_relative(
            source, center, height_m, attitude_sim_quat
        )
        if relative is None:
            return self._hold(attitude_sim_quat)
        if self._estimator is not None and ray_origin is not None and obs is not None:
            # Only on a genuine reference-plane hit (ray_origin is not None): the
            # bearing-only fallback and the legacy no-mounts path give the estimator no
            # ground-plane geometry to turn pixel uncertainty into a position fix
            # ("estimate range only where the observation geometry supports it").
            self._estimator.update(
                t, velocity_ned, ray_origin, ray_direction, height_m, self._reference_height_m,
                self.intrinsics["fx"], self.intrinsics["fy"], obs.uncertainty_px,
                clipped=obs.clipped, confidence=obs.confidence, sightline_unit=sightline,
            )
        return self._steer(
            t, attitude_sim_quat, velocity_ned, source, center, side, sightline, relative,
            discontinuous=self._tracker.discontinuous,
        )

    def _camera_signal(
        self, camera: str, sees: bool, obs: Optional[TargetObservation]
    ) -> CameraSignal:
        """This tick's `tracking.CameraSignal` for `camera`.

        With structured detections (Gazebo/mounts path), confidence comes from
        `pixels.detect_target`'s own score and uncertainty from the per-camera
        associator's current track sigma (whether or not *this* tick detected
        anything: a stale, growing sigma is exactly the "runaway prediction
        uncertainty" the state machine's uncertainty bound exists to catch). The
        legacy no-mounts path has neither: a visible camera counts as maximally
        confident (so the documented upward/forward priority still governs
        selection) and uncertainty never bounds it.
        """
        if self._associators is None:
            return CameraSignal(visible=sees, confidence=1.0 if sees else 0.0)
        confidence = obs.confidence if (obs is not None and sees) else 0.0
        track = self._associators[camera].track
        uncertainty_px = None if track is None else track.sigma_px
        return CameraSignal(visible=sees, confidence=confidence, uncertainty_px=uncertainty_px)

    def _detect(self, camera: str, image: Optional[np.ndarray], t: float) -> Optional[TargetObservation]:
        """`TargetObservation` through this camera's associator, or None on the legacy
        (no-mounts) path, which keeps `target_square`/`target_pixel_count` instead.

        `allow_reacquire=True`: a target that moved further between ticks than the gate
        currently allows must still be picked up (no persistent distractor is assumed
        just because the gate briefly misses); a genuine same-frame distractor is still
        rejected because the gated candidate is tried, and accepted, first.
        """
        if self._associators is None:
            return None
        return self._associators[camera].update(image, t, camera, allow_reacquire=True)

    def _tracked_square(self, image: Optional[np.ndarray], obs: Optional[TargetObservation]):
        """(center, side) of the tracked square for this tick, or (None, None).

        With structured detections (Gazebo/mounts path), the center and side come from
        the associator-gated `TargetObservation`, not every red-like pixel in the frame.
        Without them (legacy no-mounts path), `target_square` is unchanged.
        """
        if obs is not None:
            if obs.status is not DetectionStatus.DETECTED:
                return None, None
            u_min, v_min, u_max, v_max = obs.bounds
            side = float(max(u_max - u_min, v_max - v_min) + 1)
            return np.array(obs.reference, dtype=float), side
        if image is None:
            return None, None
        square = target_square(image)
        if square is None:
            return None, None
        return np.array([square[0], square[1]], dtype=float), float(square[2])

    def reset_derivative_history(self) -> None:
        """Clear the proportional-navigation derivative history (previous sightline/time)
        and any per-camera detection track or target-motion estimator state.

        Call this after a `VehicleState` clock discontinuity (sim-time reset, backward
        jump, or detected pause) is reported: the previous sightline/timestamp pair was
        recorded before the discontinuity, and feeding it into the next line-of-sight-rate
        computation would fabricate a spurious angular rate from two readings that are no
        longer actually adjacent in (sim) time. The same discontinuity also makes any
        standing per-camera detection track (Gazebo/mounts path) a stale prediction of
        where the target now is, and the target-motion estimator's own filter time/state
        meaningless (its `dt`s would span the discontinuity); both are dropped rather
        than built on across it.
        """
        self._last_sightline = None
        self._last_image_time = None
        if self._associators is not None:
            for associator in self._associators.values():
                associator.reset()
        if self._estimator is not None:
            self._estimator.reset()
        self._tracker.force_discontinuity()

    def complete(self, reason: Optional[str] = None) -> None:
        """External "the flight is over" signal (the truth-side scorer, through the
        tracker/runner): never set from inside this module, and guidance itself never
        reads truth. One-way: a call after `ABORT` is a no-op, so a late completion
        signal cannot paper over an already-declared failure.
        """
        self._tracker.complete(reason)

    def _sees(self, image: Optional[np.ndarray], obs: Optional[TargetObservation] = None) -> bool:
        if obs is not None:
            return obs.status is DetectionStatus.DETECTED
        count = 0 if image is None else target_pixel_count(image)
        return count >= self.lock_pixels

    def _hold(self, attitude_sim_quat: np.ndarray) -> HomingCommand:
        """Camera tick that is not steering: zero acceleration, and not a permanent loss."""
        self._tracking = False
        self._acceleration_ned = np.zeros(3)
        self._source = None
        self._tracker_center = None
        self._tracker_side = None
        self._requested_velocity_ned = None
        self._limiting_reasons = []
        if not self._locked:
            self._velocity_ned = np.array(self._initial_velocity_ned, dtype=float, copy=True)
        return self._command(attitude_sim_quat)

    def _terminal_command(self, attitude_sim_quat: np.ndarray) -> HomingCommand:
        """`ABORT`/`COMPLETE`: stop commanding, instead of holding a stale velocity
        indefinitely. The documented, configurable decision (see `HomingCommand`) is
        that a caller maps `ABORT` to `SetpointStream.mark_fatal()`; this zeroed
        velocity is a safety fallback for any caller that does not check
        `tracking_state` and still forwards the command as-is.
        """
        self._tracking = False
        self._acceleration_ned = np.zeros(3)
        self._velocity_ned = np.zeros(3)
        self._source = None
        self._tracker_center = None
        self._tracker_side = None
        self._requested_velocity_ned = None
        self._limiting_reasons = []
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
        discontinuous: bool = False,
    ) -> HomingCommand:
        if discontinuous or self._last_image_time is None:
            previous_sightline = None
            dt = 0.0
        else:
            previous_sightline = self._last_sightline
            dt = float(t) - float(self._last_image_time)
        accel = pn_acceleration(
            self.nav_constant,
            velocity_ned,
            sightline,
            previous_sightline,
            dt,
        )
        new_yaw = float(np.arctan2(sightline[1], sightline[0]))
        if self._envelope is None:
            # Legacy no-mounts path: unchanged (every existing legacy test asserts
            # this exact direction-based velocity/accel output).
            self._acceleration_ned = accel
            self._velocity_ned = desired_velocity(relative, self._component_cap)
            self._yaw_rad = new_yaw
        else:
            # Gazebo (mounts) path (plan Phase 7): one coherent velocity/acceleration/
            # yaw command, PN folded into the *velocity* (so it demonstrably changes
            # the transmitted command even though `acceleration_ned` stays masked by
            # default -- see `homing_config.PROPOSED_ENABLE_ACCELERATION_FEEDFORWARD`),
            # then bounded by the vehicle envelope and continuity limits.
            reasons: List[str] = []
            aim_point = relative
            target_motion_now = None
            if self._estimator is not None:
                target_motion_now = self._estimator.predict_to(
                    self._last_control_time, velocity_ned, sightline_unit=sightline
                )
            if (
                target_motion_now is not None
                and target_motion_now.status is TargetMotionStatus.OBSERVABLE
                and target_motion_now.target_velocity_ned is not None
            ):
                # Fix round 1 (Task 10): `pn_lead_point` advances the aim point along
                # the TARGET's own estimated ground velocity, not the vehicle-relative
                # rate (`relative_velocity_ned` = target velocity minus vehicle
                # velocity -- nonzero for a stationary target whenever the vehicle is
                # moving, per `target_estimator.TargetMotionEstimate`'s own
                # docstring). Every test exercising this branch before Task 10 primed
                # the estimator with a stationary vehicle, where the two fields are
                # numerically identical, masking the bug (see
                # `tests/test_homing_vehicle_envelope.py`'s nonzero-vehicle-velocity
                # PN-lead tests).
                aim_point = pn_lead_point(
                    self.nav_constant, relative, target_motion_now.target_velocity_ned, self._pn_lead_time_s
                )
            elif target_motion_now is not None:
                reasons.append(
                    "target_motion_unavailable:" + (target_motion_now.reason or target_motion_now.status.value)
                )
            requested_velocity = max_feasible_velocity(aim_point, self._envelope.max_speed_m_s)
            # Fix round 1 (coherent single command path): the acceleration REQUEST
            # must describe the same motion as the velocity request above, not an
            # independently-derived PN sightline-rotation-rate quantity (`accel`,
            # still computed above for the legacy path) that can disagree in sign and
            # magnitude with the actual commanded velocity trend -- reviewer-verified
            # via the moving-target estimator (OBSERVABLE) case. It is instead the
            # finite-difference slew this tick's velocity request would imply,
            # relative to the previously TRANSMITTED (already-limited) velocity; it
            # still goes through `limit_command`'s own jerk/accel-magnitude continuity
            # below like any other request, so the returned `acceleration_ned` stays a
            # bounded, continuous command, just one that is actually consistent with
            # `velocity_ned`'s own tick-to-tick change (see
            # `tests/test_homing_vehicle_envelope.py`'s
            # `test_acceleration_ned_is_consistent_with_the_transmitted_velocity_command`).
            if dt > 0.0:
                requested_acceleration = (requested_velocity - self._velocity_ned) / dt
            else:
                requested_acceleration = np.zeros(3)
            limited = limit_command(
                requested_velocity, requested_acceleration, new_yaw,
                self._velocity_ned, self._acceleration_ned, self._yaw_rad,
                dt, self._envelope,
                center_px=center, image_width_px=self.intrinsics.get("width"),
                image_height_px=self.intrinsics.get("height"),
            )
            self._requested_velocity_ned = requested_velocity
            self._velocity_ned = limited.velocity_ned
            self._acceleration_ned = limited.acceleration_ned
            self._yaw_rad = limited.yaw_rad
            self._limiting_reasons = reasons + list(limited.reasons)
        self._tracking = True
        self._last_sightline = sightline / np.linalg.norm(sightline)
        self._last_image_time = float(t)
        self._source = source
        self._tracker_center = np.array(center, dtype=float, copy=True)
        self._tracker_side = float(side)
        return self._command(attitude_sim_quat)

    def _command(self, attitude_sim_quat: np.ndarray) -> HomingCommand:
        velocity = np.array(self._velocity_ned, dtype=float, copy=True)
        if self._mounts is None:
            body_velocity = body_frd_velocity(velocity, attitude_sim_quat)
        else:
            rotation = Rotation.from_quat(attitude_homing_to_ned_frd(attitude_sim_quat))
            body_velocity = rotation.inv().apply(velocity)
        target_motion = None
        if self._estimator is not None and self._last_control_time is not None:
            target_motion = self._estimator.predict_to(
                self._last_control_time, self._last_velocity_ned, sightline_unit=self._last_sightline
            )
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
            self._detections,
            self._tracker.state,
            self._tracker.abort_reason,
            target_motion,
            self._requested_velocity_ned,
            self._limiting_reasons,
        )
