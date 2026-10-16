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
