"""Camera-ray conversion for the Gazebo cameras: NED/FRD transforms, translation, rendering."""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.frames import attitude_enu_flu_to_ned_frd, enu_to_ned
from rotorpy.guidance.camera_geometry import (
    CameraMount,
    mount_from_gazebo_pose,
    pixel_ray_ned,
    pinhole_intrinsics,
    project_ned_point,
)
from rotorpy.guidance.gazebo_bridge import attitude_gazebo_to_homing, attitude_homing_to_ned_frd
from rotorpy.guidance.homing import _sightline_ned
from rotorpy.guidance.camera_geometry import pitched_extrinsics
from rotorpy.guidance.homing_config import gazebo_camera_mounts, load_homing_config
from rotorpy.sensors.camera import PinholeCamera

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "examples" / "camera_homing.jsonc"
INTRINSICS = pinhole_intrinsics(2236, 2236, 90.0, 90.0)

FORWARD_POSE = ((0.36, 0.0, 0.05), (0.0, 0.698131701, 0.0))
UPWARD_POSE = ((0.0, 0.0, 0.05), (0.0, -0.698131701, 0.0))


@pytest.fixture(scope="module")
def mounts():
    return gazebo_camera_mounts(load_homing_config(CONFIG_PATH))


def test_forward_mount_is_ahead_and_above_the_body_in_frd(mounts):
    forward, _ = mounts
    assert np.allclose(forward.position_frd_m, [0.36, 0.0, -0.05])
    pitch = np.deg2rad(40.0)
    assert np.allclose(forward.rotation_frd_from_camera @ [0, 0, 1], [np.cos(pitch), 0.0, np.sin(pitch)])
    assert np.allclose(forward.rotation_frd_from_camera @ [1, 0, 0], [0.0, 1.0, 0.0])
    assert np.allclose(forward.rotation_frd_from_camera @ [0, 1, 0], [-np.sin(pitch), 0.0, np.cos(pitch)])


def test_upward_mount_looks_forty_degrees_above_the_nose(mounts):
    _, upward = mounts
    assert np.allclose(upward.position_frd_m, [0.0, 0.0, -0.05])
    pitch = np.deg2rad(-40.0)
    assert np.allclose(upward.rotation_frd_from_camera @ [0, 0, 1], [np.cos(pitch), 0.0, np.sin(pitch)])
    assert upward.rotation_frd_from_camera[2, 2] < 0.0


@pytest.mark.parametrize("pose", [FORWARD_POSE, UPWARD_POSE])
def test_gazebo_sdf_pose_matches_the_configured_mount(pose, mounts):
    built = mount_from_gazebo_pose(*pose)
    configured = mounts[0] if pose is FORWARD_POSE else mounts[1]
    assert np.allclose(built.position_frd_m, configured.position_frd_m, atol=1e-8)
    assert np.allclose(built.rotation_frd_from_camera, configured.rotation_frd_from_camera, atol=1e-8)


def test_mount_rpy_uses_the_sdf_rotation_order():
    roll, pitch, yaw = 0.2, -0.3, 0.5
    rx = Rotation.from_euler("x", roll).as_matrix()
    ry = Rotation.from_euler("y", pitch).as_matrix()
    rz = Rotation.from_euler("z", yaw).as_matrix()
    sensor_in_flu = rz @ ry @ rx
    optical_in_flu = np.column_stack([-sensor_in_flu[:, 1], -sensor_in_flu[:, 2], sensor_in_flu[:, 0]])
    mount = mount_from_gazebo_pose((0.1, 0.2, 0.3), (roll, pitch, yaw))
    assert np.allclose(mount.rotation_frd_from_camera, np.diag([1.0, -1.0, -1.0]) @ optical_in_flu)
    assert np.allclose(mount.position_frd_m, [0.1, -0.2, -0.3])


def test_mount_rotation_is_a_proper_rotation(mounts):
    for mount in mounts:
        assert isinstance(mount, CameraMount)
        assert np.allclose(mount.rotation_frd_from_camera.T @ mount.rotation_frd_from_camera, np.eye(3))
        assert np.isclose(np.linalg.det(mount.rotation_frd_from_camera), 1.0)


@pytest.mark.parametrize("seed", range(6))
def test_homing_attitude_adapter_agrees_with_the_frames_module(seed):
    rotation = Rotation.random(random_state=seed)
    legacy = attitude_gazebo_to_homing(rotation.as_quat())
    adapted = Rotation.from_quat(attitude_homing_to_ned_frd(legacy))
    reference = Rotation.from_quat(attitude_enu_flu_to_ned_frd(rotation.as_quat()))
    assert np.allclose(adapted.as_matrix(), reference.as_matrix(), atol=1e-9)


def test_pixel_right_of_center_is_east_for_a_north_facing_vehicle(mounts):
    north_facing_ned = np.eye(3)
    origin, direction = pixel_ray_ned(1118.0 + 223.6, 1118.0, INTRINSICS, mounts[0], north_facing_ned)
    pitch = np.deg2rad(40.0)
    expected = np.array([np.cos(pitch), 0.2, np.sin(pitch)]) / np.sqrt(1.04)
    assert direction[1] > 0.0
    assert np.allclose(direction, expected, atol=1e-6)
    assert np.allclose(origin, [0.36, 0.0, -0.05])


def test_pixel_above_center_looks_up_and_below_looks_down(mounts):
    level = np.eye(3)
    _, above = pixel_ray_ned(1118.0, 1118.0 - 400.0, INTRINSICS, mounts[0], level)
    _, below = pixel_ray_ned(1118.0, 1118.0 + 400.0, INTRINSICS, mounts[0], level)
    _, center = pixel_ray_ned(1118.0, 1118.0, INTRINSICS, mounts[0], level)
    assert above[2] < center[2] < below[2]


def test_rays_are_unit_length_and_origin_rotates_with_the_body(mounts):
    yaw_east = Rotation.from_euler("z", 90.0, degrees=True).as_matrix()
    origin, direction = pixel_ray_ned(900.0, 700.0, INTRINSICS, mounts[0], yaw_east)
    assert np.isclose(np.linalg.norm(direction), 1.0)
    assert np.allclose(origin, [0.0, 0.36, -0.05])


def _random_pose(rng):
    rotation = Rotation.random(random_state=rng.integers(1 << 30))
    position = rng.uniform(-50.0, 50.0, 3)
    return position, rotation.as_matrix()


@pytest.mark.parametrize("camera", [0, 1])
@pytest.mark.parametrize("seed", range(8))
def test_known_points_project_and_reconstruct_through_both_cameras(mounts, camera, seed):
    rng = np.random.default_rng(seed)
    mount = mounts[camera]
    checked = 0
    for _ in range(200):
        position, rotation = _random_pose(rng)
        # A point in front of the camera, inside its frustum, at 20 to 200 m.
        u, v = rng.uniform(100.0, 2136.0, 2)
        depth = rng.uniform(20.0, 200.0)
        _, ray = pixel_ray_ned(u, v, INTRINSICS, mount, rotation)
        origin = position + rotation @ mount.position_frd_m
        point = origin + depth * ray
        projected = project_ned_point(point, position, rotation, INTRINSICS, mount)
        assert projected is not None
        assert np.allclose(projected, [u, v], atol=1e-6)
        ray_origin, ray_direction = pixel_ray_ned(projected[0], projected[1], INTRINSICS, mount, rotation)
        offset = point - (position + ray_origin)
        distance = np.linalg.norm(offset - (offset @ ray_direction) * ray_direction)
        assert distance < 1e-6
        checked += 1
    assert checked == 200


def test_points_behind_the_camera_do_not_project(mounts):
    behind = np.array([-50.0, 0.0, 0.0])
    assert project_ned_point(behind, np.zeros(3), np.eye(3), INTRINSICS, mounts[0]) is None


def test_camera_translation_changes_the_ray_origin_and_the_projection(mounts):
    point = np.array([30.0, 4.0, 3.0])
    with_offset = project_ned_point(point, np.zeros(3), np.eye(3), INTRINSICS, mounts[0])
    no_offset = project_ned_point(
        point, np.zeros(3), np.eye(3), INTRINSICS,
        CameraMount(np.zeros(3), mounts[0].rotation_frd_from_camera),
    )
    assert not np.allclose(with_offset, no_offset, atol=1.0)


class _PointWorld(object):
    """The world interface PinholeCamera renders: a cloud of red features and no blocks."""

    def __init__(self, points_enu):
        self.points = np.asarray(points_enu, dtype=float)

    def get_surface_features(self):
        return self.points

    def get_feature_colors(self):
        return np.tile([1.0, 0.0, 0.0], (len(self.points), 1))

    def get_feature_descriptors(self):
        return None

    def get_block_bounding_boxes(self):
        return []


def _render_enu_flu(position_enu, gazebo_quat, target_enu, pose):
    """Render in RotorPy's native ENU world and FLU body, with Gazebo's own sensor pose."""
    (px, py, pz), (roll, pitch, yaw) = pose
    sensor = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    optical_in_flu = np.column_stack([-sensor[:, 1], -sensor[:, 2], sensor[:, 0]])
    camera = PinholeCamera(
        intrinsics=INTRINSICS,
        extrinsics={
            "position": np.array([px, py, pz]),
            "orientation": Rotation.from_matrix(optical_in_flu.T).as_quat(),
        },
        splat_radius=6,
        feature_output="rgb",
    )
    cluster = np.asarray(target_enu) + np.stack(np.meshgrid(*[np.linspace(-0.25, 0.25, 6)] * 3), -1).reshape(-1, 3)
    image = camera.render(
        _PointWorld(cluster), {"x": np.asarray(position_enu, float), "q": gazebo_quat}, background_color=[0.0, 0.0, 0.0]
    )["image"]
    return image


def _centroid(image):
    rows, columns = np.nonzero(image[:, :, 0] > 0.5)
    assert rows.size > 0, "the target must be visible"
    return float(np.mean(columns)), float(np.mean(rows))


def _distance_to_ray(point, origin, direction):
    offset = point - origin
    return float(np.linalg.norm(offset - (offset @ direction) * direction))


@pytest.mark.parametrize("heading_deg", [0.0, 26.57, 50.0, -10.0])
@pytest.mark.parametrize("attitude_deg", [(0.0, 0.0), (5.0, -4.0), (-6.0, 8.0)])
def test_forward_rendered_image_reconstructs_through_the_known_point(mounts, heading_deg, attitude_deg):
    position_enu = np.array([1.0, -2.0, 30.0])
    target_enu = np.array([50.0, 100.0, 1.5])
    rotation = Rotation.from_euler("ZYX", [90.0 - heading_deg, attitude_deg[1], attitude_deg[0]], degrees=True)
    image = _render_enu_flu(position_enu, rotation.as_quat(), target_enu, FORWARD_POSE)
    u, v = _centroid(image)
    r_ned_frd = Rotation.from_quat(attitude_enu_flu_to_ned_frd(rotation.as_quat())).as_matrix()
    offset, direction = pixel_ray_ned(u, v, INTRINSICS, mounts[0], r_ned_frd)
    origin = enu_to_ned(position_enu) + offset
    assert _distance_to_ray(enu_to_ned(target_enu), origin, direction) < 0.2


def test_upward_rendered_image_reconstructs_through_the_known_point(mounts):
    position_enu = np.array([0.0, 0.0, 12.0])
    target_enu = np.array([3.0, 40.0, 1.5])
    # A steep dive puts the 70 degree upward camera on the ground target.
    rotation = Rotation.from_euler("ZYX", [90.0, 70.0, 0.0], degrees=True)
    image = _render_enu_flu(position_enu, rotation.as_quat(), target_enu, UPWARD_POSE)
    u, v = _centroid(image)
    r_ned_frd = Rotation.from_quat(attitude_enu_flu_to_ned_frd(rotation.as_quat())).as_matrix()
    offset, direction = pixel_ray_ned(u, v, INTRINSICS, mounts[1], r_ned_frd)
    origin = enu_to_ned(position_enu) + offset
    assert _distance_to_ray(enu_to_ned(target_enu), origin, direction) < 0.2


def test_legacy_rotorpy_sightline_keeps_its_mirrored_convention():
    # The Python example renders in a world whose body +y points east at the identity yaw,
    # so a pixel right of center is west there. That convention stays behind _sightline_ned.
    sightline = _sightline_ned(np.array([1118.0 + 223.6, 1118.0]), np.array([0, 0, 0, 1.0]), INTRINSICS,
                               pitched_extrinsics(10.0))
    assert sightline[1] < 0.0


def test_guidance_steers_at_the_visible_centre_not_the_ground_contact(mounts):
    from rotorpy.guidance.homing_config import guidance_from_config

    config = load_homing_config(CONFIG_PATH)
    position_enu = np.array([0.0, 0.0, 30.0])
    target_enu = np.array([50.0, 100.0, 1.5])
    rotation = Rotation.from_euler("ZYX", [90.0 - 26.565, 0.0, 0.0], degrees=True)
    image = _render_enu_flu(position_enu, rotation.as_quat(), target_enu, FORWARD_POSE)
    guidance = guidance_from_config(config)
    command = guidance.update(
        1.0, 30.0, attitude_gazebo_to_homing(rotation.as_quat()), np.zeros(3), image, None
    )
    assert command.tracking
    unit = command.velocity_ned / np.linalg.norm(command.velocity_ned)

    def angle_to(point_ned):
        wanted = point_ned - enu_to_ned(position_enu)
        return np.degrees(np.arccos(np.clip(unit @ wanted / np.linalg.norm(wanted), -1.0, 1.0)))

    assert angle_to(enu_to_ned(target_enu)) < 0.1
    assert angle_to(np.array([100.0, 50.0, 0.0])) > 0.3
