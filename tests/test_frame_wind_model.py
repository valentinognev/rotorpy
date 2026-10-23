import copy

import numpy as np
import pytest

from scipy.spatial.transform import Rotation

from rotorpy.estimators.wind_ekf import WindEKF
from rotorpy.estimators.wind_ukf import WindUKF
from rotorpy.frames import FrameConvention, attitude_enu_flu_to_ned_frd
from rotorpy.vehicles.crazyflie_params import quad_params


def test_ukf_frd_level_hover_is_identity():
    """FRD Euler angles are relative to the identity quaternion.

    Subtracting attitude_enu_flu_to_ned_frd(identity) reports yaw -pi/2 at
    the identity hover, and reports zero yaw at that converted heading.
    """
    filt = WindUKF(copy.deepcopy(quad_params), dt=0.01)
    filt.set_frame(FrameConvention.FRD_NED)
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    level = filt._tilt_orientation(identity)
    assert level.as_euler("zyx") == pytest.approx([0.0, 0.0, 0.0])

    pitched = Rotation.from_euler("y", 0.1).as_quat()
    assert filt._tilt_orientation(pitched).as_euler("zyx") == pytest.approx([0.0, 0.1, 0.0])

    trim = attitude_enu_flu_to_ned_frd(identity)
    assert filt._tilt_orientation(trim).magnitude() == pytest.approx(np.pi / 2.0)


def test_ukf_frd_hover_body_z_accel_is_zero():
    params = copy.deepcopy(quad_params)
    filt = WindUKF(params, dt=0.01)
    filt.set_frame(FrameConvention.FRD_NED)
    x = np.zeros(9)
    filt.uk[0] = -9.81
    x_next = filt.f(x, 0.01)
    assert x_next[5] == pytest.approx(0.0, abs=1e-6)


def test_ukf_flu_hover_body_z_accel_is_zero():
    params = copy.deepcopy(quad_params)
    filt = WindUKF(params, dt=0.01)
    filt.set_frame(FrameConvention.FLU_ENU)
    assert filt.g == 9.81
    x = np.zeros(9)
    filt.uk[0] = 9.81
    x_next = filt.f(x, 0.01)
    assert x_next[5] == pytest.approx(0.0, abs=1e-6)


def test_ukf_set_frame_twice_keeps_frd_hover():
    params = copy.deepcopy(quad_params)
    filt = WindUKF(params, dt=0.01)
    filt.set_frame(FrameConvention.FRD_NED)
    filt.set_frame(FrameConvention.FRD_NED)
    assert filt.g == 9.81
    x = np.zeros(9)
    filt.uk[0] = -9.81
    x_next = filt.f(x, 0.01)
    assert x_next[5] == pytest.approx(0.0, abs=1e-6)


def test_ukf_pitch_and_roll_gravity_couplings_follow_frame():
    params = copy.deepcopy(quad_params)
    dt = 0.01
    pitched = np.zeros(9)
    pitched[1] = 0.1
    rolled = np.zeros(9)
    rolled[2] = 0.1

    flu = WindUKF(params, dt=dt)
    flu.set_frame(FrameConvention.FLU_ENU)
    flu.uk[0] = 9.81
    frd = WindUKF(copy.deepcopy(quad_params), dt=dt)
    frd.set_frame(FrameConvention.FRD_NED)
    frd.uk[0] = -9.81

    flu_pitch = flu.f(pitched, dt)
    frd_pitch = frd.f(pitched, dt)
    flu_roll = flu.f(rolled, dt)
    frd_roll = frd.f(rolled, dt)

    assert flu_pitch[3] == pytest.approx(0.01 * 9.81 * 0.1, abs=1e-6)
    assert frd_pitch[3] == pytest.approx(-0.01 * 9.81 * 0.1, abs=1e-6)
    assert flu_roll[4] == pytest.approx(-0.01 * 9.81 * 0.1, abs=1e-6)
    assert frd_roll[4] == pytest.approx(0.01 * 9.81 * 0.1, abs=1e-6)
    assert flu_pitch[5] == pytest.approx(0.0, abs=1e-6)
    assert frd_pitch[5] == pytest.approx(0.0, abs=1e-6)


def test_commanded_hover_thrust_cancels_gravity_in_both_frames():
    params = copy.deepcopy(quad_params)
    state = {"w": np.zeros(3)}
    command = {"cmd_thrust": params["mass"] * 9.81}
    x = np.zeros(9)

    ukf = WindUKF(params, dt=0.01)
    ukf.set_frame(FrameConvention.FRD_NED)
    ukf.uk = ukf.construct_control_vector(state, command)
    assert ukf.uk[0] == pytest.approx(-9.81)
    assert ukf.f(x, 0.01)[5] == pytest.approx(0.0, abs=1e-6)
    ukf.set_frame(FrameConvention.FLU_ENU)
    ukf.uk = ukf.construct_control_vector(state, command)
    assert ukf.g == 9.81
    assert ukf.uk[0] == pytest.approx(9.81)
    assert ukf.f(x, 0.01)[5] == pytest.approx(0.0, abs=1e-6)

    ekf = WindEKF(copy.deepcopy(quad_params), dt=0.01)
    ekf.set_frame(FrameConvention.FRD_NED)
    uk_frd = ekf.construct_control_vector(state, command)
    assert uk_frd[0] == pytest.approx(-9.81)
    assert ekf.process_model(x, uk_frd)[5] == pytest.approx(0.0, abs=1e-6)
    ekf.set_frame(FrameConvention.FLU_ENU)
    uk_flu = ekf.construct_control_vector(state, command)
    assert uk_flu[0] == pytest.approx(9.81)
    assert ekf.process_model(x, uk_flu)[5] == pytest.approx(0.0, abs=1e-6)


def test_ekf_frd_hover_body_z_accel_is_zero():
    params = copy.deepcopy(quad_params)
    filt = WindEKF(params, dt=0.01)
    filt.set_frame(FrameConvention.FRD_NED)
    filt.set_frame(FrameConvention.FRD_NED)
    assert filt.g == 9.81
    x = np.zeros(9)
    uk = np.array([-9.81, 0.0, 0.0, 0.0])
    x_next = filt.process_model(x, uk)
    assert x_next[5] == pytest.approx(0.0, abs=1e-6)


def test_ekf_flu_hover_body_z_accel_is_zero():
    params = copy.deepcopy(quad_params)
    filt = WindEKF(params, dt=0.01)
    filt.set_frame(FrameConvention.FLU_ENU)
    assert filt.g == 9.81
    x = np.zeros(9)
    uk = np.array([9.81, 0.0, 0.0, 0.0])
    x_next = filt.process_model(x, uk)
    assert x_next[5] == pytest.approx(0.0, abs=1e-6)


def test_ekf_pitch_and_roll_gravity_couplings_follow_frame():
    params = copy.deepcopy(quad_params)
    pitched = np.zeros(9)
    pitched[1] = 0.1
    rolled = np.zeros(9)
    rolled[0] = 0.1
    uk_flu = np.array([9.81, 0.0, 0.0, 0.0])
    uk_frd = np.array([-9.81, 0.0, 0.0, 0.0])

    flu = WindEKF(params, dt=0.01)
    flu.set_frame(FrameConvention.FLU_ENU)
    frd = WindEKF(copy.deepcopy(quad_params), dt=0.01)
    frd.set_frame(FrameConvention.FRD_NED)

    assert flu.process_model(pitched, uk_flu)[3] == pytest.approx(0.01 * 9.81 * 0.1, abs=1e-6)
    assert frd.process_model(pitched, uk_frd)[3] == pytest.approx(-0.01 * 9.81 * 0.1, abs=1e-6)
    assert flu.process_model(rolled, uk_flu)[4] == pytest.approx(-0.01 * 9.81 * 0.1, abs=1e-6)
    assert frd.process_model(rolled, uk_frd)[4] == pytest.approx(0.01 * 9.81 * 0.1, abs=1e-6)


def test_body_x_accel_uses_pitch_rate_not_roll_rate():
    """Body-x acceleration uses vertical speed times pitch rate.

    Roll rate 0.4 with pitch rate 0 leaves body-x unchanged. Pitch rate
    0.4 with body-z speed 2 changes body-x by -0.008 over 0.01 s.
    The EKF Jacobian column for body-z speed is that same pitch rate.
    """
    params = copy.deepcopy(quad_params)
    x = np.zeros(9)
    x[5] = 2.0
    dt = 0.01
    pitch = np.array([9.81, 0.0, 0.4, 0.0])
    roll = np.array([9.81, 0.4, 0.0, 0.0])

    ekf = WindEKF(params, dt=dt)
    assert ekf.process_model(x, pitch)[3] == pytest.approx(-0.008)
    assert ekf.process_model(x, roll)[3] == pytest.approx(0.0)
    ekf.computeJacobians(x, pitch)
    assert ekf.A[3, 5] == pytest.approx(-0.4)
    ekf.computeJacobians(x, roll)
    assert ekf.A[3, 5] == pytest.approx(0.0)

    ukf = WindUKF(copy.deepcopy(quad_params), dt=dt)
    ukf.uk = pitch
    assert ukf.f(x, dt)[3] == pytest.approx(-0.008)
    ukf.uk = roll
    assert ukf.f(x, dt)[3] == pytest.approx(0.0)


def test_ekf_jacobian_at_hover_matches_process_model():
    params = copy.deepcopy(quad_params)
    flu = WindEKF(params, dt=0.01)
    flu.set_frame(FrameConvention.FLU_ENU)
    frd = WindEKF(copy.deepcopy(quad_params), dt=0.01)
    frd.set_frame(FrameConvention.FRD_NED)

    x = np.zeros(9)
    uk_flu = np.array([9.81, 0.0, 0.0, 0.0])
    uk_frd = np.array([-9.81, 0.0, 0.0, 0.0])
    flu.computeJacobians(x, uk_flu)
    frd.computeJacobians(x, uk_frd)

    assert flu.A[3, 1] == pytest.approx(9.81)
    assert flu.A[4, 0] == pytest.approx(-9.81)
    assert frd.A[3, 1] == pytest.approx(-9.81)
    assert frd.A[4, 0] == pytest.approx(9.81)

    _assert_angle_step_matches_jacobian(flu, x, uk_flu)
    _assert_angle_step_matches_jacobian(frd, x, uk_frd)


def _assert_angle_step_matches_jacobian(filt, x, uk):
    x_next = filt.process_model(x, uk)
    assert x_next[5] == pytest.approx(0.0, abs=1e-6)
    eps = 1e-6
    for index in (0, 1, 2):
        dx = np.zeros(9)
        dx[index] = eps
        nonlinear = filt.process_model(x + dx, uk)
        linear = x_next + dx + filt.dt * filt.A.dot(dx)
        assert nonlinear == pytest.approx(linear, abs=1e-8)
