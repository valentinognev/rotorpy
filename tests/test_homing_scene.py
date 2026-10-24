import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from rotorpy.guidance.pixels import target_pixel_count
from rotorpy.guidance.scene import hover_rotor_speed, initial_velocity_ned, nominal_camera, nominal_initial_state, nominal_upward_camera, nominal_world
from rotorpy.vehicles.hummingbird_params import quad_params

def test_initial_course_and_hover_thrust():
    velocity = initial_velocity_ned(np.array([0.0, 0.0, -30.0]), np.array([100.0, 50.0, 0.0]), 10.0)
    assert velocity[2] == 0.0
    assert np.isclose(np.linalg.norm(velocity), 10.0)
    assert velocity[0] > 0.0 and velocity[1] > 0.0
    omega = hover_rotor_speed(quad_params)
    assert np.isclose(4.0 * quad_params["k_eta"] * omega**2, quad_params["mass"] * 9.81)

def test_nominal_start_sees_a_stable_target_of_at_least_36_pixels():
    world = nominal_world()
    camera = nominal_camera()
    state = nominal_initial_state()
    first = target_pixel_count(camera.measurement(state, world)["image"])
    second = target_pixel_count(camera.measurement(state, world)["image"])
    assert first == second
    assert first >= 36
    assert np.allclose(state["x"], [0.0, 0.0, 30.0])

def test_upward_camera_matches_the_forward_lens_and_looks_up():
    forward = nominal_camera()
    upward = nominal_upward_camera()
    assert upward.intrinsics == forward.intrinsics
    assert np.allclose(upward.extrinsics["position"], 0.0)
    body_to_camera = Rotation.from_quat(upward.extrinsics["orientation"]).as_matrix()
    optical_axis = body_to_camera[2]
    assert optical_axis[2] > optical_axis[0] > 0.0
    assert optical_axis[1] == pytest.approx(0.0)
