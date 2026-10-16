import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.guidance.camera_geometry import (
    boresight_frd, pinhole_intrinsics, pitched_extrinsics, pixel_to_camera_ray, square_resolution)
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.world import World

def test_five_megapixel_square_and_focal_length():
    assert square_resolution() == (2236, 2236)
    intrinsics = pinhole_intrinsics(2236, 2236, 90.0, 90.0)
    assert np.isclose(intrinsics["fx"], 1118.0)
    assert np.isclose(intrinsics["fy"], 1118.0)
    assert np.allclose(intrinsics["dist_coeffs"], 0.0)

def test_boresight_is_ten_degrees_down_in_frd():
    pitch = np.deg2rad(10.0)
    assert np.allclose(boresight_frd(10.0), [np.cos(pitch), 0.0, np.sin(pitch)])

def test_principal_point_ray_is_camera_z():
    intrinsics = pinhole_intrinsics(32, 32, 90.0, 90.0)
    ray = pixel_to_camera_ray(intrinsics["cx"], intrinsics["cy"], intrinsics)
    assert np.allclose(ray, [0.0, 0.0, 1.0])

def test_level_body_projects_boresight_to_the_principal_point():
    intrinsics = pinhole_intrinsics(32, 32, 90.0, 90.0)
    extrinsics = pitched_extrinsics(10.0)
    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics, splat_radius=1, feature_output="rgb")
    pitch = np.deg2rad(10.0)
    point = 10.0 * np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    world = World({
        "bounds": {"extents": (-20, 20, -20, 20, -20, 20)},
        "blocks": [],
        "features": {"points": [point.tolist()], "colors": [[1.0, 0.0, 0.0]]},
    })
    state = {"x": np.zeros(3), "q": np.array([0.0, 0.0, 0.0, 1.0])}
    measurement = camera.measurement(state, world)
    assert measurement["keypoints"].shape[0] == 1
    assert np.allclose(measurement["keypoints"][0], [intrinsics["cx"], intrinsics["cy"]], atol=1.0)
    assert np.allclose(extrinsics["position"], 0.0)
    Rotation.from_quat(extrinsics["orientation"])  # quaternion parses
