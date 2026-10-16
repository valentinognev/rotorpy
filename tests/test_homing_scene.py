import numpy as np
from rotorpy.guidance.pixels import target_pixel_count
from rotorpy.guidance.scene import hover_rotor_speed, initial_velocity_ned, nominal_camera, nominal_initial_state, nominal_world
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
