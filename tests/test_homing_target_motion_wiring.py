"""Target-motion estimator wiring into `HomingGuidance` (plan Phase 6).

The estimator only runs on the Gazebo (mounts) path, where `pixel_ray_ned` gives the
true NED ray geometry it needs; the legacy no-mounts path is untouched. Wiring is
purely additive: `HomingCommand.target_motion` is populated, but nothing in the
existing velocity/acceleration/yaw steering law reads it yet (Task 7 owns that).
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import boresight_frd, pitched_extrinsics
from rotorpy.guidance.gazebo_bridge import attitude_gazebo_to_homing
from rotorpy.guidance.homing import HomingCommand, HomingGuidance
from rotorpy.guidance.homing_config import guidance_from_config, load_homing_config
from rotorpy.guidance.speed import desired_velocity
from rotorpy.guidance.target_estimator import TargetMotionEstimate, TargetMotionStatus
from rotorpy.guidance.telemetry import target_motion_fields_from_command

CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"
NORTH_FACING = Rotation.from_euler("z", 90.0, degrees=True)
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
INTRINSICS = {"fx": 16.0, "fy": 16.0, "width": 32, "height": 32,
              "cx": 15.5, "cy": 15.5, "dist_coeffs": np.zeros(5)}


@pytest.fixture(scope="module")
def config():
    return load_homing_config(CONFIG_PATH)


def _intrinsics(config):
    width, height = int(config["image_width_px"]), int(config["image_height_px"])
    fx = (width / 2.0) / np.tan(np.deg2rad(config["fov_x_deg"]) / 2.0)
    fy = (height / 2.0) / np.tan(np.deg2rad(config["fov_y_deg"]) / 2.0)
    return fx, fy, width / 2.0, height / 2.0


def _blank(config):
    width, height = int(config["image_width_px"]), int(config["image_height_px"])
    return np.zeros((height, width, 3), dtype=np.float32)


def _solid_blob(config, column, row, half=6):
    image = _blank(config)
    r, c = int(round(row)), int(round(column))
    image[r - half:r + half + 1, c - half:c + half + 1, 0] = 1.0
    return image


def _course():
    direction = np.array([100.0, 50.0, 0.0])
    return 10.0 * direction / np.linalg.norm(direction)


def _blob(row0, col0):
    image = np.zeros((32, 32, 3))
    image[row0:row0 + 6, col0:col0 + 6, 0] = 1.0
    return image


def test_legacy_no_mounts_path_never_builds_target_motion():
    """The legacy (no-mounts) Python-example path has no estimator at all: it has no
    NED ray geometry to feed, so `target_motion` must stay `None` on every command."""
    guidance = HomingGuidance(3.0, 36, _course(), INTRINSICS, pitched_extrinsics(10.0), pitched_extrinsics(-70))
    command = guidance.update(0.0, 30.0, IDENTITY, np.array([10.0, 5.0, 0.0]), _blob(13, 13), None)
    assert command.target_motion is None
    command = guidance.update(0.05, 30.0, IDENTITY, np.array([10.0, 5.0, 0.0]), _blob(13, 13), None)
    assert command.target_motion is None


def test_mounts_path_populates_an_observable_target_motion_estimate_once_tracking(config):
    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    guidance = guidance_from_config(config)
    _, _, cx, cy = _intrinsics(config)
    command = guidance.update(0.0, 30.0, attitude, np.zeros(3), _solid_blob(config, cx, cy), None)
    assert command.tracking and command.locked
    assert isinstance(command.target_motion, TargetMotionEstimate)
    assert command.target_motion.status is TargetMotionStatus.OBSERVABLE
    assert command.target_motion.relative_position_ned is not None


def test_mounts_path_target_motion_does_not_change_the_existing_steering_output(config):
    """Wiring in the estimator must be purely additive: the velocity/acceleration/yaw
    command is still only `desired_velocity`/`pn_acceleration` on the tracked relative
    vector, unaffected by whatever the estimator independently computes."""
    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    guidance = guidance_from_config(config)
    _, _, cx, cy = _intrinsics(config)
    command = guidance.update(0.0, 30.0, attitude, np.zeros(3), _solid_blob(config, cx + 50.0, cy), None)
    assert command.tracking
    # No acceleration on the very first tracked tick (no previous sightline yet) and the
    # velocity is exactly the capped relative-position-derived speed law: this must stay
    # true whether or not `target_motion` is wired in.
    assert np.allclose(command.acceleration_ned, 0.0)
    assert np.linalg.norm(command.velocity_ned) <= float(config["speed_component_cap_m_s"]) * np.sqrt(3.0) + 1e-6


def test_mounts_path_target_motion_stays_none_shaped_until_the_geometry_supports_it(config):
    """Before any lock, the estimator has never been fed a usable ray: unobservable."""
    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    guidance = guidance_from_config(config)
    command = guidance.update(0.0, 30.0, attitude, np.zeros(3), _blank(config), _blank(config))
    assert command.target_motion is not None
    assert command.target_motion.status is TargetMotionStatus.UNOBSERVABLE


def test_target_motion_fields_from_command_is_empty_without_an_estimate():
    command = HomingCommand(
        np.zeros(3), np.zeros(3), np.zeros(3), 0.0, True, True, "forward",
        np.array([4.0, 4.0]), 6.0,
    )
    assert target_motion_fields_from_command(command) == {}


def test_target_motion_fields_from_command_reports_the_estimate():
    estimate = TargetMotionEstimate(
        status=TargetMotionStatus.OBSERVABLE,
        reason=None,
        relative_position_ned=np.array([10.0, 5.0, 3.0]),
        relative_velocity_ned=np.array([0.1, -0.2, 0.0]),
        target_velocity_ned=np.array([0.1, -0.2, 0.0]),
        closing_speed_m_s=1.5,
        position_sigma_m=0.2,
        velocity_sigma_m_s=0.3,
        age_s=0.05,
    )
    command = HomingCommand(
        np.zeros(3), np.zeros(3), np.zeros(3), 0.0, True, True, "forward",
        np.array([4.0, 4.0]), 6.0, target_motion=estimate,
    )
    fields = target_motion_fields_from_command(command)
    motion = fields["target_motion"]
    assert motion["status"] == "observable"
    assert motion["relative_position_ned"] == [10.0, 5.0, 3.0]
    assert motion["closing_speed_m_s"] == 1.5
    assert motion["age_s"] == 0.05
