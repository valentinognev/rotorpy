import numpy as np
from rotorpy.guidance.speed import desired_velocity, max_feasible_velocity

def test_axis_aligned_speed_is_19():
    assert np.allclose(desired_velocity(np.array([2.0, 0.0, 0.0])), [19.0, 0.0, 0.0])
    assert np.allclose(desired_velocity(np.array([0.0, 0.0, -5.0])), [0.0, 0.0, -19.0])

def test_equal_north_and_down_components_are_each_19():
    velocity = desired_velocity(np.array([1.0, 0.0, 1.0]))
    assert np.allclose(velocity, [19.0, 0.0, 19.0])
    assert np.isclose(np.linalg.norm(velocity), 19.0 * np.sqrt(2.0))

def test_zero_direction():
    assert np.allclose(desired_velocity(np.zeros(3)), 0.0)

def test_max_feasible_velocity_is_a_total_speed_norm_objective_not_per_component():
    """Phase 7: the maximum-feasible-speed objective is norm-based, unlike the legacy
    per-component `desired_velocity` (whose diagonal case is sqrt(2) times faster)."""
    velocity = max_feasible_velocity(np.array([1.0, 0.0, 1.0]), 19.0)
    assert np.isclose(np.linalg.norm(velocity), 19.0)
    assert np.allclose(velocity, [19.0 / np.sqrt(2.0), 0.0, 19.0 / np.sqrt(2.0)])

def test_max_feasible_velocity_preserves_direction():
    direction = np.array([3.0, -4.0, 0.0])
    velocity = max_feasible_velocity(direction, 10.0)
    assert np.isclose(np.linalg.norm(velocity), 10.0)
    assert np.allclose(velocity / np.linalg.norm(velocity), direction / np.linalg.norm(direction))

def test_max_feasible_velocity_zero_direction_is_zero():
    assert np.allclose(max_feasible_velocity(np.zeros(3), 19.0), 0.0)
