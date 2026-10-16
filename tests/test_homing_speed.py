import numpy as np
from rotorpy.guidance.speed import desired_velocity

def test_axis_aligned_speed_is_19():
    assert np.allclose(desired_velocity(np.array([2.0, 0.0, 0.0])), [19.0, 0.0, 0.0])
    assert np.allclose(desired_velocity(np.array([0.0, 0.0, -5.0])), [0.0, 0.0, -19.0])

def test_equal_north_and_down_components_are_each_19():
    velocity = desired_velocity(np.array([1.0, 0.0, 1.0]))
    assert np.allclose(velocity, [19.0, 0.0, 19.0])
    assert np.isclose(np.linalg.norm(velocity), 19.0 * np.sqrt(2.0))

def test_zero_direction():
    assert np.allclose(desired_velocity(np.zeros(3)), 0.0)
