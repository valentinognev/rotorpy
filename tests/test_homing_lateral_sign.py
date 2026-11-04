"""Gazebo camera-ray geometry: lateral sign, and no target position in guidance.

The oracle is written in Gazebo's own ENU world and FLU body frames, so it
shares no frame algebra with the NED/FRD implementation under test.
"""

import copy
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.guidance.gazebo_bridge import attitude_gazebo_to_homing
from rotorpy.guidance.homing_config import guidance_from_config, load_homing_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"
# Gazebo sdf: forward camera 0.36 m ahead and 0.05 m above the body, pitched 40 deg down.
FORWARD = {"position_flu": np.array([0.36, 0.0, 0.05]), "pitch_deg": 40.0}
# Upward camera is raised 0.05 m above the body origin (clears the propeller disks from
# its 40 deg-above-the-nose FOV), pitched 40 deg above the nose.
UPWARD = {"position_flu": np.array([0.0, 0.0, 0.05]), "pitch_deg": -40.0}
NORTH_FACING = Rotation.from_euler("z", 90.0, degrees=True)


@pytest.fixture(scope="module")
def config():
    return load_homing_config(CONFIG_PATH)


def _intrinsics(config):
    width, height = int(config["image_width_px"]), int(config["image_height_px"])
    fx = (width / 2.0) / np.tan(np.deg2rad(config["fov_x_deg"]) / 2.0)
    fy = (height / 2.0) / np.tan(np.deg2rad(config["fov_y_deg"]) / 2.0)
    return fx, fy, width / 2.0, height / 2.0


def _blob_image(config, column, row, half=6):
    image = np.zeros((int(config["image_height_px"]), int(config["image_width_px"]), 3), dtype=np.float32)
    r, c = int(round(row)), int(round(column))
    image[r - half:r + half + 1, c - half:c + half + 1, 0] = 1.0
    return image


def _bearing_oracle(config, gazebo_rotation, camera, center, height_m):
    """NED bearing of the body-to-target sightline, derived in ENU/FLU."""
    fx, fy, cx, cy = _intrinsics(config)
    ray_optical = np.array([(center[0] - cx) / fx, (center[1] - cy) / fy, 1.0])
    sensor = Rotation.from_euler("y", camera["pitch_deg"], degrees=True).as_matrix()
    # Optical x is right = -left, y is down = -up, z is forward.
    optical_in_flu = np.column_stack([-sensor[:, 1], -sensor[:, 2], sensor[:, 0]])
    matrix = gazebo_rotation.as_matrix()
    ray_enu = matrix @ (optical_in_flu @ ray_optical)
    offset_enu = matrix @ camera["position_flu"]
    assert ray_enu[2] < 0.0, "the pixel must look below the horizon"
    scale = -(height_m + offset_enu[2]) / ray_enu[2]
    relative_enu = offset_enu + scale * ray_enu
    return float(np.arctan2(relative_enu[0], relative_enu[1]))


def _observe(config, gazebo_rotation, column, row, height_m=30.0, camera="forward", velocity=None):
    guidance = guidance_from_config(config)
    image = _blob_image(config, column, row)
    attitude = attitude_gazebo_to_homing(gazebo_rotation.as_quat())
    velocity = np.zeros(3) if velocity is None else velocity
    forward, upward = (image, None) if camera == "forward" else (None, image)
    return guidance.update(1.0, height_m, attitude, velocity, forward, upward)


def test_pixel_right_of_the_principal_point_steers_east_when_facing_north(config):
    _, _, cx, cy = _intrinsics(config)
    command = _observe(config, NORTH_FACING, cx + 223.6, cy)
    assert command.tracking and command.source == "forward"
    oracle = _bearing_oracle(config, NORTH_FACING, FORWARD, command.tracker_center, 30.0)
    assert oracle > 0.0
    # The buggy path produced the mirrored bearing, 2 * atan(223.6 / fx) = 22.62 deg away.
    assert command.yaw_rad > 0.0
    assert np.degrees(abs(command.yaw_rad - oracle)) < 0.05


def test_pixel_left_of_the_principal_point_steers_west_when_facing_north(config):
    _, _, cx, cy = _intrinsics(config)
    command = _observe(config, NORTH_FACING, cx - 223.6, cy)
    assert command.yaw_rad < 0.0
    oracle = _bearing_oracle(config, NORTH_FACING, FORWARD, command.tracker_center, 30.0)
    assert np.degrees(abs(command.yaw_rad - oracle)) < 0.05


def test_centered_observation_points_along_the_nose(config):
    _, _, cx, cy = _intrinsics(config)
    command = _observe(config, NORTH_FACING, cx, cy)
    assert abs(np.degrees(command.yaw_rad)) < 0.05


@pytest.mark.parametrize("row_offset", [-150.0, 150.0])
def test_above_and_below_center_keep_the_heading(config, row_offset):
    _, _, cx, cy = _intrinsics(config)
    command = _observe(config, NORTH_FACING, cx, cy + row_offset, height_m=60.0)
    assert abs(np.degrees(command.yaw_rad)) < 0.05


@pytest.mark.parametrize(
    "euler_deg",
    [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 90.0),
        (0.0, 0.0, 200.0),
        (8.0, -6.0, 35.0),
        (-12.0, 15.0, 300.0),
    ],
)
@pytest.mark.parametrize("offset", [(223.6, 0.0), (-350.0, 120.0), (500.0, 60.0)])
def test_forward_camera_bearing_matches_the_enu_flu_oracle(config, euler_deg, offset):
    rotation = Rotation.from_euler("xyz", euler_deg, degrees=True)
    _, _, cx, cy = _intrinsics(config)
    command = _observe(config, rotation, cx + offset[0], cy + offset[1], height_m=40.0)
    assert command.tracking
    oracle = _bearing_oracle(config, rotation, FORWARD, command.tracker_center, 40.0)
    error = np.arctan2(np.sin(command.yaw_rad - oracle), np.cos(command.yaw_rad - oracle))
    assert np.degrees(abs(error)) < 0.05


def test_upward_camera_bearing_matches_the_oracle_for_a_diving_vehicle(config):
    rotation = Rotation.from_euler("xyz", (0.0, 65.0, 120.0), degrees=True)
    _, _, cx, cy = _intrinsics(config)
    command = _observe(config, rotation, cx + 250.0, cy + 300.0, height_m=40.0, camera="upward")
    assert command.tracking and command.source == "upward"
    oracle = _bearing_oracle(config, rotation, UPWARD, command.tracker_center, 40.0)
    error = np.arctan2(np.sin(command.yaw_rad - oracle), np.cos(command.yaw_rad - oracle))
    assert np.degrees(abs(error)) < 0.05


def _moved_target(config):
    moved = copy.deepcopy(config)
    moved["target_ground_ned"] = [-80.0, 200.0, 0.0]
    moved["box_center_ned"] = [-80.0, 200.0, -0.5]
    moved["box_extents_m"] = [-80.25, -79.75, 199.75, 200.25, 0.0, 1.0]
    return moved


def _command_fields(command):
    return (
        command.velocity_ned, command.acceleration_ned, command.velocity_body_frd,
        command.yaw_rad, command.locked, command.tracking, command.source,
    )


def test_guidance_output_is_independent_of_the_true_target_position(config):
    moved = _moved_target(config)
    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    _, _, cx, cy = _intrinsics(config)
    image = _blob_image(config, cx + 120.0, cy + 40.0)
    first, second = guidance_from_config(config), guidance_from_config(moved)
    for step, pictures in enumerate([(None, None), (None, None), (image, None), (None, None), (image, None)]):
        args = (0.05 * step, 30.0, attitude, np.array([4.0, 2.0, 0.0]), *pictures)
        for a, b in zip(_command_fields(first.update(*args)), _command_fields(second.update(*args))):
            assert np.array_equal(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("heading_deg", [0.0, 90.0, 215.0])
def test_course_before_lock_follows_the_vehicle_heading_not_the_target(config, heading_deg):
    # NED heading is clockwise from north. The Gazebo yaw about up is 90 - heading.
    rotation = Rotation.from_euler("z", 90.0 - heading_deg, degrees=True)
    attitude = attitude_gazebo_to_homing(rotation.as_quat())
    command = guidance_from_config(config).update(0.0, 30.0, attitude, np.zeros(3), None, None)
    speed = float(config["initial_speed_m_s"])
    heading = np.radians(heading_deg)
    assert np.allclose(command.velocity_ned, [speed * np.cos(heading), speed * np.sin(heading), 0.0], atol=1e-9)
    assert np.isclose(np.cos(command.yaw_rad - heading), 1.0)


def test_locked_guidance_keeps_steering_at_and_below_the_reference_height(config):
    """Once locked, fresh pixels at/below the 1.5 m visible-centre reference height must
    still move the command: no frozen replay of the last pre-descent velocity/acceleration.

    The upward mount's translation offset (0.05 m) is negligible next to the reference
    height, so height_m == reference_height_m (1.5 m) is effectively the degenerate
    plane-intersection case `reference_relative_position` cannot solve; smaller heights
    (1.0, 0.5 m) stay degenerate too. The diving attitude keeps the
    upward-camera ray aimed below the horizon throughout (a level attitude would make the
    upward camera look at the sky, a separate, legitimate no-bearing case). Each step uses a
    very different pixel position, so a correct fix tracks it (a bug freezes velocity_ned
    byte-for-byte from the lock, with tracking flipped to False).
    """
    guidance = guidance_from_config(config)
    lock_attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    diving = attitude_gazebo_to_homing(Rotation.from_euler("xyz", (0.0, 65.0, 120.0), degrees=True).as_quat())
    _, _, cx, cy = _intrinsics(config)
    velocity = np.zeros(3)

    lock = guidance.update(0.0, 30.0, lock_attitude, velocity, _blob_image(config, cx + 100.0, cy), None)
    assert lock.locked and lock.tracking

    at_reference = guidance.update(
        1.0, 1.5, diving, velocity, None, _blob_image(config, cx + 250.0, cy + 300.0)
    )
    assert at_reference.tracking is True, "a visible, fresh pixel must not flip tracking off"
    assert not np.allclose(at_reference.velocity_ned, lock.velocity_ned)
    assert np.all(np.isfinite(at_reference.velocity_ned))

    below_a = guidance.update(2.0, 1.0, diving, velocity, None, _blob_image(config, cx - 280.0, cy + 250.0))
    assert below_a.tracking is True
    assert not np.allclose(below_a.velocity_ned, at_reference.velocity_ned)

    below_b = guidance.update(3.0, 0.5, diving, velocity, None, _blob_image(config, cx + 350.0, cy + 200.0))
    assert below_b.tracking is True
    assert not np.allclose(below_b.velocity_ned, below_a.velocity_ned)
    assert below_b.locked is True
