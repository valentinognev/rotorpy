"""One authoritative scene: config, sdf, launch script, replay and ROS CameraInfo agree."""

import copy
from pathlib import Path

import numpy as np
import pytest

from rotorpy.guidance.homing_config import (
    gazebo_scene_from_config,
    gazebo_target_reference,
    load_homing_config,
)
from rotorpy.guidance.scene import box_target_reference
from rotorpy.guidance.scene_consistency import (
    check_repo_scene_consistency,
    check_scene_consistency,
    config_intrinsics,
    parse_camera_sdf,
    parse_cylinder_sdf,
    parse_launch_spawns,
    validate_camera_info,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "examples" / "camera_homing.jsonc"
VEHICLE_SDF = ROOT / "gazeboModels" / "hummingbird" / "hummingbird.sdf.jinja"
TARGET_SDF = ROOT / "gazeboModels" / "homing" / "target_cylinder.sdf"
LAUNCH_SCRIPT = ROOT / "Dockerfiles" / "sitl_hummingbird_run.sh"


@pytest.fixture
def config():
    return load_homing_config(CONFIG_PATH)


def _camera_info(config, **changes):
    """What gazebo_ros_camera publishes for the configured lens (1-based principal point)."""
    width, height = int(config["image_width_px"]), int(config["image_height_px"])
    fx = (width / 2.0) / np.tan(np.deg2rad(config["fov_x_deg"]) / 2.0)
    fy = (height / 2.0) / np.tan(np.deg2rad(config["fov_y_deg"]) / 2.0)
    cx, cy = (width + 1.0) / 2.0, (height + 1.0) / 2.0
    info = {
        "width": width,
        "height": height,
        "distortion_model": "plumb_bob",
        "D": [0.0] * 5,
        "K": [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        "P": [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
    }
    info.update(changes)
    return info


def test_config_intrinsics_follow_the_configured_lens(config):
    intrinsics = config_intrinsics(config)
    assert intrinsics["width"] == 2236 and intrinsics["height"] == 2236
    assert np.isclose(intrinsics["fx"], 1118.0)
    assert np.isclose(intrinsics["cx"], 1118.0)


def test_matching_camera_info_validates(config):
    assert validate_camera_info(config, _camera_info(config)) == []


def test_camera_info_with_other_field_of_view_is_rejected(config):
    wider = dict(config, fov_x_deg=100.0, fov_y_deg=100.0)
    problems = validate_camera_info(config, _camera_info(wider))
    assert any("fx" in problem for problem in problems)
    assert any("fy" in problem for problem in problems)


def test_camera_info_size_distortion_and_projection_matrix_are_checked(config):
    info = _camera_info(config, width=640, D=[0.1, 0.0, 0.0, 0.0, 0.0])
    problems = validate_camera_info(config, info)
    assert any("width" in problem for problem in problems)
    assert any("distortion" in problem.lower() for problem in problems)
    info = _camera_info(config)
    info["P"][0] += 50.0
    assert any("P" in problem for problem in validate_camera_info(config, info))


def test_camera_info_principal_point_offset_beyond_tolerance_is_rejected(config):
    info = _camera_info(config)
    info["K"][2] += 20.0
    assert any("cx" in problem for problem in validate_camera_info(config, info))


def test_camera_info_must_be_complete_and_finite(config):
    assert validate_camera_info(config, {"width": 2236, "height": 2236})
    info = _camera_info(config)
    info["K"][0] = float("nan")
    assert validate_camera_info(config, info)


def test_repo_scene_files_agree_with_the_config(config):
    assert check_repo_scene_consistency(config, ROOT) == []


def test_parsers_read_the_spawn_and_model_values():
    cameras = parse_camera_sdf(VEHICLE_SDF.read_text())
    assert set(cameras) == {"forward", "upward"}
    forward = cameras["forward"]
    assert np.allclose(forward["position_flu_m"], [0.36, 0.0, 0.05])
    assert np.isclose(forward["rpy_rad"][1], np.deg2rad(10.0), atol=1e-8)
    assert forward["width"] == 2236 and forward["update_rate_hz"] == 20.0
    assert np.isclose(cameras["upward"]["rpy_rad"][1], np.deg2rad(-70.0), atol=1e-8)
    cylinder = parse_cylinder_sdf(TARGET_SDF.read_text())
    assert cylinder == {"radius_m": 0.75, "length_m": 3.0, "visual_radius_m": 0.75, "visual_length_m": 3.0}
    spawns = parse_launch_spawns(LAUNCH_SCRIPT.read_text())
    assert spawns["target_enu"] == (50.0, 100.0, 1.5)
    assert spawns["vehicle_enu"] == (0.0, 0.0, 30.0)
    assert np.isclose(spawns["vehicle_yaw_enu_rad"], np.arctan2(100.0, 50.0))


def _texts():
    return VEHICLE_SDF.read_text(), TARGET_SDF.read_text(), LAUNCH_SCRIPT.read_text()


def _mismatches(config, vehicle=None, target=None, launch=None):
    texts = _texts()
    return check_scene_consistency(
        config, vehicle or texts[0], target or texts[1], launch or texts[2]
    )


def test_changed_field_of_view_in_the_config_is_reported(config):
    problems = _mismatches(dict(config, fov_x_deg=95.0))
    assert any("fov" in problem.lower() for problem in problems)


def test_changed_forward_camera_position_in_the_config_is_reported(config):
    problems = _mismatches(dict(config, forward_camera_position_flu_m=[0.30, 0.0, 0.05]))
    assert any("forward" in problem and "position" in problem for problem in problems)


def test_changed_camera_pitch_in_the_sdf_is_reported(config):
    vehicle = VEHICLE_SDF.read_text().replace("0.174532925", "0.3")
    assert any("forward" in problem and "pitch" in problem for problem in _mismatches(config, vehicle=vehicle))


def test_changed_camera_rate_is_reported(config):
    assert any("rate" in problem for problem in _mismatches(dict(config, camera_rate_hz=10)))


def test_cylinder_centre_spawned_on_the_ground_is_reported(config):
    launch = LAUNCH_SCRIPT.read_text().replace("-x 50 -y 100 -z 1.5", "-x 50 -y 100 -z 0")
    problems = _mismatches(config, launch=launch)
    assert any("target spawn" in problem for problem in problems)


def test_changed_cylinder_size_is_reported(config):
    target = TARGET_SDF.read_text().replace("<length>3.0</length>", "<length>2.0</length>")
    assert any("length" in problem for problem in _mismatches(config, target=target))


def test_changed_vehicle_spawn_height_or_heading_is_reported(config):
    launch = LAUNCH_SCRIPT.read_text().replace("-z 30 -Y", "-z 20 -Y")
    assert any("vehicle spawn" in problem for problem in _mismatches(config, launch=launch))
    launch = LAUNCH_SCRIPT.read_text().replace("math.atan2(100.0, 50.0)", "math.atan2(50.0, 100.0)")
    assert any("yaw" in problem for problem in _mismatches(config, launch=launch))


def test_changed_takeoff_ground_spawn_height_is_reported(config):
    # Minor (e), fix round 1: the airborne-spawn line ("-z 30 -Y", checked above by
    # `test_changed_vehicle_spawn_height_or_heading_is_reported`) was the only
    # hummingbird_0 spawn the consistency check ever looked at -- the ground spawn used
    # in takeoff mode ("-x 0 -y 0 -z 0.2 -Y", only taken when
    # HOMING_STARTUP_MODE=takeoff) was never checked against
    # `PROPOSED_TAKEOFF_SPAWN_HEIGHT_M`.
    launch = LAUNCH_SCRIPT.read_text().replace("-x 0 -y 0 -z 0.2 -Y", "-x 0 -y 0 -z 5.0 -Y")
    assert any("takeoff" in problem.lower() for problem in _mismatches(config, launch=launch))


def test_changed_takeoff_ground_spawn_horizontal_position_is_reported(config):
    launch = LAUNCH_SCRIPT.read_text().replace("-x 0 -y 0 -z 0.2 -Y", "-x 3 -y 0 -z 0.2 -Y")
    assert any("takeoff" in problem.lower() for problem in _mismatches(config, launch=launch))


def test_target_reference_separates_the_visible_centre_from_the_ground_contact(config):
    reference = gazebo_target_reference(config)
    assert np.allclose(reference.ground_contact_ned, [100.0, 50.0, 0.0])
    assert np.allclose(reference.visible_reference_ned, [100.0, 50.0, -1.5])
    assert reference.reference_height_m == 1.5
    assert reference.radius_m == 0.75 and reference.height_m == 3.0
    assert not np.allclose(reference.ground_contact_ned, reference.visible_reference_ned)


def test_python_example_box_reference_is_its_scored_centre(config):
    reference = box_target_reference(config)
    assert np.allclose(reference.visible_reference_ned, config["box_center_ned"])
    assert np.allclose(reference.ground_contact_ned, config["target_ground_ned"])


def test_replay_world_matches_the_spawned_cylinder(config):
    world, forward, upward = gazebo_scene_from_config(config)
    cylinder = parse_cylinder_sdf(TARGET_SDF.read_text())
    spawn = parse_launch_spawns(LAUNCH_SCRIPT.read_text())["target_enu"]
    east, north, centre_up = spawn
    radius, length = cylinder["radius_m"], cylinder["length_m"]
    # The replay world is north, east, up. Its block is the cylinder's bounding box.
    expected = [
        north - radius, north + radius, east - radius, east + radius,
        centre_up - length / 2.0, centre_up + length / 2.0,
    ]
    assert np.allclose(world.get_block_bounding_boxes()[0], expected)
    assert np.allclose(forward.extrinsics["position"], [0.36, 0.0, 0.05])
    assert np.allclose(upward.extrinsics["position"], [0.0, 0.0, 0.0])


def test_replay_scene_is_independent_of_the_python_example_box(config):
    changed = copy.deepcopy(config)
    changed["box_extents_m"] = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
    world, _, _ = gazebo_scene_from_config(changed)
    reference, _, _ = gazebo_scene_from_config(config)
    assert np.allclose(world.get_block_bounding_boxes()[0], reference.get_block_bounding_boxes()[0])
