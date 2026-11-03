import numpy as np
from rotorpy.guidance.navigation import pn_acceleration

def test_first_sample_and_zero_dt_are_zero():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([4.0, 0.0, 0.0])
    assert np.allclose(pn_acceleration(3.0, v, u, None, 0.5), 0.0)
    assert np.allclose(pn_acceleration(3.0, v, u, np.array([0.0, 1.0, 0.0]), 0.0), 0.0)

def test_known_acceleration_scales_with_n_and_closing_speed():
    u = np.array([1.0, 0.0, 0.0])
    u_prev = np.array([0.0, 1.0, 0.0])
    v = np.array([4.0, 0.0, 0.0])
    accel = pn_acceleration(3.0, v, u, u_prev, 0.5)
    # N * Vc * (omega × u) is -east. The brief's +24 is u × omega.
    assert np.allclose(accel, [0.0, -24.0, 0.0])
    assert np.allclose(pn_acceleration(6.0, v, u, u_prev, 0.5), [0.0, -48.0, 0.0])
    assert np.allclose(pn_acceleration(3.0, [8.0, 0.0, 0.0], u, u_prev, 0.5), [0.0, -48.0, 0.0])
    assert np.dot(accel, u) == 0.0

def test_zero_closing_speed_is_zero_acceleration():
    accel = pn_acceleration(3.0, np.array([0.0, 5.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0.5)
    assert np.allclose(accel, 0.0)


def test_pn_lead_point_is_the_relative_position_with_stationary_target():
    from rotorpy.guidance.navigation import pn_lead_point
    relative = np.array([10.0, 5.0, 0.0])
    assert np.allclose(pn_lead_point(3.0, relative, np.zeros(3), 0.1), relative)


def test_pn_lead_point_scales_linearly_with_the_navigation_constant():
    """Phase 7 PN integration: changing the navigation constant must demonstrably
    change the aim point whenever the target's own estimated motion is nonzero."""
    from rotorpy.guidance.navigation import pn_lead_point
    relative = np.array([10.0, 5.0, 0.0])
    relative_velocity = np.array([1.0, 0.0, 0.0])
    lead_time = 0.2
    near = pn_lead_point(2.0, relative, relative_velocity, lead_time)
    far = pn_lead_point(5.0, relative, relative_velocity, lead_time)
    assert not np.allclose(near, far)
    assert np.allclose(far - near, (5.0 - 2.0) * lead_time * relative_velocity)
