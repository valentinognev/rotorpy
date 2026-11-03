"""JSONC scene file for the camera-homing example."""

import json
from pathlib import Path

import numpy as np

from rotorpy.guidance.camera_geometry import mount_from_pitch
from rotorpy.guidance.homing import HomingGuidance
from rotorpy.guidance.origin_alignment import EstimatorOriginAlignment
from rotorpy.guidance.scene import (
    TargetReference,
    nominal_camera,
    nominal_initial_state,
    nominal_upward_camera,
    nominal_world,
)
from rotorpy.guidance.scene_consistency import config_intrinsics

CONTROLLERS = ("geometric", "pid")

# Proposed, not verified: truth miss distance that counts as reaching the target.
PROPOSED_REACH_RADIUS_M = 1.5

# Proposed, not verified: below this horizontal speed the measured velocity has no usable
# heading, so the pre-lock course falls back to the vehicle's attitude instead.
PROPOSED_MIN_COURSE_SPEED_M_S = 1.0

# Proposed, not verified: wall-time staleness timeouts for the tracker's comms watchdogs
# (`watchdog.CommsWatchdogs`). Each link going stale this long with no fresh message is
# treated as a comms failure independent of simulation time. Heartbeat is 3.0 s because
# PX4's ~1 Hz period means a 1.0 s timeout would latch on jitter.
PROPOSED_HEARTBEAT_TIMEOUT_S = 3.0
PROPOSED_TELEMETRY_TIMEOUT_S = 0.5
PROPOSED_IMAGE_TIMEOUT_S = 1.0

# Proposed, not verified: `tracking.TrackingStateMachine` thresholds for the recoverable
# tracking state machine (plan Phase 5). 1 consecutive frame to (re)acquire keeps the
# single-sighting lock/steer behavior every existing test already exercises; raise either
# frame count to require sustained confirmation before steering/resuming instead. The
# switch margin only matters between two simultaneously visible, equal-priority cameras
# (the Gazebo forward/upward pair has a fixed priority, so it rarely engages there; it
# protects a future same-priority pair from flapping on a marginal confidence lead). The
# prediction gap and uncertainty bounds are independent: whichever is tighter triggers
# the explicit ABORT failure response first (never an indefinitely held stale command).
PROPOSED_ACQUIRE_FRAMES = 1
PROPOSED_REACQUIRE_FRAMES = 1
PROPOSED_CAMERA_SWITCH_MARGIN = 0.1
PROPOSED_MAX_PREDICTION_GAP_S = 2.0
PROPOSED_MAX_TRACK_UNCERTAINTY_PX = 150.0


# Proposed, not verified: `target_estimator.TargetMotionEstimator`'s bounded ground-
# target-motion envelope and filter thresholds (plan Phase 6). The envelope is ground
# motion only (a standing/moving ground target, not an aerial one): 5 m/s is a brisk
# walk/jog, 2 m/s^2 a generous acceleration for a person or small ground vehicle. An
# estimate whose implied speed exceeds this is flagged UNCERTAIN rather than trusted,
# the same way age or position uncertainty are. The outlier chi-squared threshold is
# the standard 2-degrees-of-freedom 99% value. `max_age_s`/`max_position_sigma_m`
# bound how long a gap or how poor a fix may be before the estimate is no longer
# treated as exact. `clipped_inflation`/`low_confidence_inflation`/`confidence_floor`
# further inflate the per-tick measurement covariance for a clipped or low-confidence
# detection (see `pixels.TargetObservation`), on top of the geometric (near-horizon)
# inflation the estimator already derives from the ray Jacobian.
PROPOSED_MAX_TARGET_SPEED_M_S = 5.0
PROPOSED_MAX_TARGET_ACCEL_M_S2 = 2.0
PROPOSED_TARGET_OUTLIER_CHI2_THRESHOLD = 9.21
PROPOSED_MAX_TARGET_AGE_S = 1.0
PROPOSED_MAX_TARGET_POSITION_SIGMA_M = 5.0
PROPOSED_TARGET_CLIPPED_INFLATION = 3.0
PROPOSED_TARGET_LOW_CONFIDENCE_INFLATION = 2.0
PROPOSED_TARGET_CONFIDENCE_FLOOR = 0.2


def target_motion_thresholds() -> dict:
    """Named target-motion-estimator thresholds, for
    `target_estimator.TargetMotionEstimator(**target_motion_thresholds())`."""
    return {
        "max_speed_m_s": PROPOSED_MAX_TARGET_SPEED_M_S,
        "max_accel_m_s2": PROPOSED_MAX_TARGET_ACCEL_M_S2,
        "outlier_chi2_threshold": PROPOSED_TARGET_OUTLIER_CHI2_THRESHOLD,
        "max_age_s": PROPOSED_MAX_TARGET_AGE_S,
        "max_position_sigma_m": PROPOSED_MAX_TARGET_POSITION_SIGMA_M,
        "clipped_inflation": PROPOSED_TARGET_CLIPPED_INFLATION,
        "low_confidence_inflation": PROPOSED_TARGET_LOW_CONFIDENCE_INFLATION,
        "confidence_floor": PROPOSED_TARGET_CONFIDENCE_FLOOR,
    }


def tracking_thresholds() -> dict:
    """Named tracking-state-machine thresholds, for
    `HomingGuidance(..., **tracking_thresholds())`."""
    return {
        "acquire_frames": PROPOSED_ACQUIRE_FRAMES,
        "reacquire_frames": PROPOSED_REACQUIRE_FRAMES,
        "camera_switch_margin": PROPOSED_CAMERA_SWITCH_MARGIN,
        "max_prediction_gap_s": PROPOSED_MAX_PREDICTION_GAP_S,
        "max_track_uncertainty_px": PROPOSED_MAX_TRACK_UNCERTAINTY_PX,
    }


def watchdog_timeouts() -> dict:
    """Named comms-watchdog timeouts, for `watchdog.CommsWatchdogs(watchdog_timeouts())`."""
    return {
        "heartbeat": PROPOSED_HEARTBEAT_TIMEOUT_S,
        "telemetry": PROPOSED_TELEMETRY_TIMEOUT_S,
        "image": PROPOSED_IMAGE_TIMEOUT_S,
    }


def strip_jsonc(text: str) -> str:
    """Drop // comments that sit outside strings."""
    out = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_homing_config(path) -> dict:
    """Parse a camera-homing JSONC file. `controller` is `geometric` or `pid`."""
    config = json.loads(strip_jsonc(Path(path).read_text()))
    controller = config["controller"]
    if controller not in CONTROLLERS:
        raise ValueError(
            "controller must be one of: {}".format(", ".join(CONTROLLERS))
        )
    if config["vehicle"] != "hummingbird":
        raise ValueError("camera homing flies the hummingbird")
    return config


def _camera_kwargs(config: dict) -> dict:
    return dict(
        width=config["image_width_px"],
        height=config["image_height_px"],
        fov_x_deg=config["fov_x_deg"],
        fov_y_deg=config["fov_y_deg"],
        frame_rate=config["camera_rate_hz"],
        splat_radius=config["splat_radius_px"],
        feature_output=config["feature_output"],
    )


def scene_from_config(config: dict):
    """Initial state, world, forward camera, and upward camera for `config`."""
    start = np.array(config["start_ned"], dtype=float)
    target = np.array(config["target_ground_ned"], dtype=float)
    speed = float(config["initial_speed_m_s"])
    position = np.array(config["camera_position_body_m"], dtype=float)
    camera = dict(_camera_kwargs(config), position=position)
    state = nominal_initial_state(
        start_ned=start,
        target_ground_ned=target,
        initial_speed=speed,
        body_rate=config["initial_body_rate_rad_s"],
        wind=config["initial_wind_m_s"],
    )
    world = nominal_world(
        bounds=config["world_bounds_m"],
        extents=config["box_extents_m"],
        color=config["box_color_rgb"],
        feature_spacing=config["feature_spacing_m"],
        feature_mode=config["feature_mode"],
        descriptor_noise=config["descriptor_noise"],
    )
    forward = nominal_camera(pitch_deg=config["forward_pitch_deg"], **camera)
    upward = nominal_upward_camera(pitch_deg=config["upward_pitch_deg"], **camera)
    return state, world, forward, upward


def gazebo_camera_mounts(config: dict):
    """Forward and upward `CameraMount`s of the Gazebo vehicle, from the config."""
    return (
        mount_from_pitch(config["forward_camera_position_flu_m"], float(config["forward_pitch_deg"])),
        mount_from_pitch(config["upward_camera_position_flu_m"], float(config["upward_pitch_deg"])),
    )


def gazebo_target_reference(config: dict) -> TargetReference:
    """The Gazebo cylinder: its ground contact, and the visible centre half its height above."""
    return TargetReference(
        ground_contact_ned=np.array(config["target_ground_ned"], dtype=float),
        height_m=float(config["gazebo_target_height_m"]),
        radius_m=float(config["gazebo_target_radius_m"]),
    )


def gazebo_target_extents_sim(config: dict) -> list:
    """Bounding box of the spawned cylinder in the replay world: north, east, up extents."""
    reference = gazebo_target_reference(config)
    north, east, down = (float(value) for value in reference.ground_contact_ned)
    radius, height = reference.radius_m, reference.height_m
    return [north - radius, north + radius, east - radius, east + radius, -down, -down + height]


def gazebo_scene_from_config(config: dict):
    """World and both cameras for replaying a Gazebo flight: the cylinder, and the real mounts.

    The Python example's block and body-origin cameras stay in `scene_from_config`.
    A sim body is (forward, left, up), so Gazebo's FLU camera positions apply as written.
    """
    world = nominal_world(
        bounds=config["world_bounds_m"],
        extents=gazebo_target_extents_sim(config),
        color=config["box_color_rgb"],
        feature_spacing=config["feature_spacing_m"],
        feature_mode=config["feature_mode"],
        descriptor_noise=config["descriptor_noise"],
    )
    camera = _camera_kwargs(config)
    forward = nominal_camera(
        pitch_deg=config["forward_pitch_deg"],
        position=np.array(config["forward_camera_position_flu_m"], dtype=float),
        **camera,
    )
    upward = nominal_upward_camera(
        pitch_deg=config["upward_pitch_deg"],
        position=np.array(config["upward_camera_position_flu_m"], dtype=float),
        **camera,
    )
    return world, forward, upward


def alignment_from_config(config: dict) -> EstimatorOriginAlignment:
    """Where the estimator's local frame sits in the world, as configured."""
    return EstimatorOriginAlignment(
        origin_world_ned=config["estimator_origin_world_ned"],
        yaw_offset_rad=np.deg2rad(float(config["estimator_yaw_offset_deg"])),
        ground_down_world_m=-float(config["ground_z_m"]),
    )


def guidance_from_config(config: dict) -> HomingGuidance:
    """The Gazebo homing law the tracker and its replay both build from `config`.

    The pre-lock course comes from the vehicle's own heading at its first
    update. No target coordinate reaches the guidance. It tracks the visible
    centre of the target, not its ground contact point.
    """
    forward_mount, upward_mount = gazebo_camera_mounts(config)
    return HomingGuidance(
        nav_constant=float(config["nav_constant"]),
        lock_pixels=int(config["lock_pixels"]),
        initial_velocity_ned=None,
        intrinsics=config_intrinsics(config),
        forward_extrinsics=None,
        upward_extrinsics=None,
        component_cap=float(config["speed_component_cap_m_s"]),
        initial_speed=float(config["initial_speed_m_s"]),
        forward_mount=forward_mount,
        upward_mount=upward_mount,
        target_reference_height_m=gazebo_target_reference(config).reference_height_m,
        min_course_speed_m_s=PROPOSED_MIN_COURSE_SPEED_M_S,
        **tracking_thresholds(),
        **target_motion_thresholds(),
    )
