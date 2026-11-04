"""Task 10 regression: the harness's synthetic cameras must render in the SAME
NED/FRD convention `homing.py`'s own Gazebo-mounts backprojection uses.

Root cause (see `task-10-report.md`): `homing_config.gazebo_scene_from_config`'s
cameras assume the standard ("simulator body", `scene.pitched_extrinsics`)
convention, which is left-right mirrored relative to `attitude_sim_quat`'s actual
"homing simulator" convention. Composed together, a rendered pixel and the
backprojection the real guidance performs on it are an exact mirror image of each
other around the image's horizontal centre once the vehicle yaws away from the one
heading every prior unit test happened to use (straight at the target): this fed the
target-motion estimator a false, growing "target velocity" for a genuinely
stationary target. `validation.true_cameras.build_true_frd_cameras`/
`world_to_true_ned` render directly in true NED/FRD instead, matching
`camera_geometry.project_ned_point`/`mount_from_gazebo_pose` by construction.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import pinhole_intrinsics, project_ned_point
from rotorpy.guidance.gazebo_bridge import attitude_homing_to_ned_frd
from rotorpy.guidance.homing_config import gazebo_camera_mounts, load_homing_config
from rotorpy.guidance.validation.true_cameras import build_true_frd_cameras, world_to_true_ned
from rotorpy.guidance.validation.vehicle_model import attitude_from_specific_force
from rotorpy.world import World
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"
_INTRINSICS = pinhole_intrinsics(2236, 2236, 90.0, 90.0)


@pytest.fixture(scope="module")
def config():
    return load_homing_config(CONFIG_PATH)


def _feature_world_ned(target_ned):
    cluster = np.stack(np.meshgrid(*[np.linspace(-0.1, 0.1, 6)] * 3), -1).reshape(-1, 3) + target_ned
    return World({
        "bounds": {"extents": [-500.0, 500.0, -500.0, 500.0, -500.0, 500.0]},
        "blocks": [],
        "features": {"points": cluster, "colors": np.ones((len(cluster), 3))},
    })


def _centroid(image):
    rows, cols = np.nonzero(image[:, :, 0] > 0.5)
    assert rows.size > 0, "the target must be visible"
    return float(np.mean(cols)), float(np.mean(rows))


@pytest.mark.parametrize("heading_deg", [26.565, 20.0, 10.0, 0.0, -10.0])
def test_forward_render_matches_the_guidance_backprojection_at_every_heading(config, heading_deg):
    """The rendered pixel of a known NED point must agree with `project_ned_point`
    (the same projection `homing.py`'s backprojection is built on) for ANY vehicle
    heading, not only the one heading (pointing straight at the target) every
    PN-lead/vehicle-envelope unit test happened to use before Task 10."""
    forward_mount, _ = gazebo_camera_mounts(config)
    forward_camera, _ = build_true_frd_cameras(dict(config, image_width_px=2236, image_height_px=2236))

    vehicle_pos_ned = np.array([30.0, 15.0, -10.0])
    target_ned = np.array([100.0, 50.0, -1.5])
    world = _feature_world_ned(target_ned)

    yaw = np.radians(heading_deg)
    attitude_sim_quat = attitude_from_specific_force(np.zeros(3), yaw)
    true_quat = attitude_homing_to_ned_frd(attitude_sim_quat)
    state = {"x": vehicle_pos_ned, "q": true_quat}
    image = forward_camera.measurement(state, world, background_color=[0.0, 0.0, 0.0])["image"]
    u_rendered, v_rendered = _centroid(image)

    rotation = Rotation.from_quat(true_quat).as_matrix()
    u_predicted, v_predicted = project_ned_point(target_ned, vehicle_pos_ned, rotation, _INTRINSICS, forward_mount)

    assert abs(u_rendered - u_predicted) < 2.0
    assert abs(v_rendered - v_predicted) < 2.0


def test_world_to_true_ned_negates_feature_and_block_z_and_swaps_block_extent_order():
    sim_world = World({
        "bounds": {"extents": [-10.0, 10.0, -10.0, 10.0, -10.0, 10.0]},
        "blocks": [{"extents": [1.0, 2.0, 3.0, 4.0, -5.0, -1.0], "color": [1.0, 0.0, 0.0]}],
        "features": {"points": np.array([[1.0, 2.0, -3.0]]), "colors": np.array([[1.0, 1.0, 1.0]])},
    })
    ned_world = world_to_true_ned(sim_world)
    assert np.allclose(ned_world.get_surface_features(), [[1.0, 2.0, 3.0]])
    boxes = ned_world.get_block_bounding_boxes()
    assert len(boxes) == 1
    xmin, xmax, ymin, ymax, zmin, zmax = boxes[0]
    assert (xmin, xmax, ymin, ymax) == (1.0, 2.0, 3.0, 4.0)
    assert (zmin, zmax) == (1.0, 5.0)
