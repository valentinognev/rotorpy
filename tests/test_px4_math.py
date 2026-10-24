import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.controllers.px4_cascade.math import (
    attitude_rate_setpoint,
    constrain_xy,
    limit_tilt,
    rate_torque,
    saturate_thrust,
    thrust_from_acceleration,
    thrust_to_attitude,
    velocity_integral,
)

def test_constrain_xy_keeps_the_priority_vector():
    assert np.allclose(constrain_xy(np.array([3.0, 0.0]), np.array([4.0, 0.0]), 12.0), [7.0, 0.0])
    assert np.allclose(constrain_xy(np.array([13.0, 0.0]), np.array([1.0, 1.0]), 12.0), [12.0, 0.0])
    assert np.allclose(constrain_xy(np.array([0.0, 0.0]), np.array([20.0, 0.0]), 12.0), [12.0, 0.0])
    assert np.allclose(constrain_xy(np.array([10.0, 0.0]), np.array([10.0, 0.0]), 12.0), [12.0, 0.0])
    scaled = constrain_xy(np.array([0.0, 9.0]), np.array([9.0, 0.0]), 12.0)
    assert np.allclose(scaled, [7.937253933193772, 9.0])
    assert np.linalg.norm(scaled) == pytest.approx(12.0)


def test_limit_tilt_stops_at_the_max_angle_and_handles_the_opposite_vector():
    g = 9.81
    ax = g * np.tan(np.deg2rad(60.0))
    raw = np.array([-ax, 0.0, g])
    raw = raw / np.linalg.norm(raw)
    limited = limit_tilt(raw, np.array([0.0, 0.0, 1.0]), np.deg2rad(45.0))
    assert np.allclose(limited, [-np.sqrt(0.5), 0.0, np.sqrt(0.5)])
    opposite = limit_tilt(np.array([0.0, 0.0, -1.0]), np.array([0.0, 0.0, 1.0]), np.deg2rad(10.0))
    assert np.allclose(opposite, [np.sin(np.deg2rad(10.0)), 0.0, np.cos(np.deg2rad(10.0))])


def test_thrust_from_acceleration_hovers_limits_tilt_and_holds_minimum_thrust():
    hover = thrust_from_acceleration(np.zeros(3), 0.5, np.deg2rad(45.0), 0.12)
    assert np.allclose(hover, [0.0, 0.0, -0.5])
    mild = thrust_from_acceleration(np.array([0.0, 0.0, 4.0]), 0.5, np.deg2rad(45.0), 0.12)
    assert np.allclose(mild, [0.0, 0.0, -0.2961264016309888])
    dropped = thrust_from_acceleration(np.array([0.0, 0.0, 76.0]), 0.5, np.deg2rad(45.0), 0.12)
    assert np.allclose(dropped, [0.0, 0.0, -0.12])
    ax = 9.81 * np.tan(np.deg2rad(60.0))
    tilted = thrust_from_acceleration(np.array([ax, 0.0, 0.0]), 0.5, np.deg2rad(45.0), 0.12)
    assert np.allclose(tilted, [0.5, 0.0, -0.5])


def test_saturate_thrust_keeps_a_horizontal_margin_then_fills_vertical():
    saturated = saturate_thrust(np.array([-0.5, -0.5, -0.9]), 1.0, 0.3)
    assert np.allclose(saturated, [-0.30822070014844877, -0.30822070014844877, -0.9])
    assert np.linalg.norm(saturated) == pytest.approx(1.0)


def test_velocity_integral_anti_windup_and_vertical_clamp():
    i_gain = np.array([0.4, 0.4, 2.0])
    free = velocity_integral(
        np.array([0.0, 0.0, 0.1]),
        np.zeros(3),
        np.array([0.0, 0.0, -0.5]),
        np.array([0.0, 0.0, -0.5]),
        np.zeros(3),
        i_gain, 1.8, 0.5, 0.12, 1.0, 0.01,
    )
    assert np.allclose(free, [0.0, 0.0, 0.002])
    stopped = velocity_integral(
        np.array([0.0, 0.0, 1.0]),
        np.zeros(3),
        np.array([0.0, 0.0, -0.12]),
        np.array([0.0, 0.0, -0.12]),
        np.zeros(3),
        i_gain, 1.8, 0.5, 0.12, 1.0, 0.01,
    )
    assert np.allclose(stopped, 0.0)
    backed = velocity_integral(
        np.array([10.0, 0.0, 0.0]),
        np.array([30.0, 0.0, 0.0]),
        np.array([0.5, 0.0, -0.5]),
        np.array([0.5, 0.0, -0.5]),
        np.zeros(3),
        i_gain, 1.8, 0.5, 0.12, 1.0, 0.01,
    )
    assert np.allclose(backed, [-0.04973333333333334, 0.0, 0.0])
    clamped = velocity_integral(
        np.array([0.0, 0.0, 10.0]),
        np.zeros(3),
        np.array([0.0, 0.0, -0.5]),
        np.array([0.0, 0.0, -0.5]),
        np.array([0.0, 0.0, 9.8]),
        i_gain, 1.8, 0.5, 0.12, 1.0, 1.0,
    )
    assert np.allclose(clamped, [0.0, 0.0, 9.81])


def test_thrust_to_attitude_builds_body_down_and_yaw():
    q, thrust_z = thrust_to_attitude(np.array([0.0, 0.0, -0.5]), 0.0)
    assert thrust_z == pytest.approx(-0.5)
    assert np.allclose(Rotation.from_quat(q).as_matrix()[:, 2], [0.0, 0.0, 1.0])
    assert Rotation.from_quat(q).as_euler("ZYX")[0] == pytest.approx(0.0)
    q_yaw, _ = thrust_to_attitude(np.array([0.0, 0.0, -0.5]), np.pi / 2.0)
    assert Rotation.from_quat(q_yaw).as_euler("ZYX")[0] == pytest.approx(np.pi / 2.0)
    q_zero, thrust_zero = thrust_to_attitude(np.zeros(3), 0.0)
    assert thrust_zero == pytest.approx(0.0)
    assert np.allclose(Rotation.from_quat(q_zero).as_matrix()[:, 2], [0.0, 0.0, 1.0])


def test_attitude_rate_mixes_yaw_handles_the_inverted_corner_and_clips():
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    limits = np.deg2rad([220.0, 220.0, 200.0])
    assert np.allclose(
        attitude_rate_setpoint(identity, identity, 0.0, 6.5, 6.5, 2.8, 0.4, limits),
        0.0,
    )
    yawed = Rotation.from_euler("z", 0.1).as_quat()
    mixed = attitude_rate_setpoint(identity, yawed, 0.0, 6.5, 6.5, 2.8, 0.4, limits)
    assert np.allclose(mixed, [0.0, 0.0, 0.2799813337066631])
    fed = attitude_rate_setpoint(identity, identity, 0.3, 6.5, 6.5, 2.8, 0.4, limits)
    assert np.allclose(fed, [0.0, 0.0, 0.3])
    inverted = np.array([0.9887710779360424, 0.14943813247359922, 0.0, 0.0])
    corner = attitude_rate_setpoint(identity, inverted, 0.0, 1.0, 1.0, 1.0, 0.4, limits)
    assert np.allclose(corner, [1.9775421558720847, 0.29887626494719843, 0.0])
    rolled = Rotation.from_rotvec([np.pi / 2.0, 0.0, 0.0]).as_quat()
    clipped = attitude_rate_setpoint(identity, rolled, 0.0, 6.5, 6.5, 2.8, 0.4, limits)
    assert clipped[0] == pytest.approx(np.deg2rad(220.0))


def test_rate_torque_integrates_unless_the_previous_mix_was_saturated():
    p = np.array([0.15, 0.15, 0.2])
    i = np.array([0.2, 0.2, 0.1])
    d = np.array([0.003, 0.003, 0.0])
    ff = np.zeros(3)
    limit = np.full(3, 0.30)
    torque, updated = rate_torque(
        np.zeros(3), np.array([0.1, 0.0, 0.0]), np.zeros(3), np.zeros(3),
        p, i, d, ff, np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), limit, 0.01,
    )
    assert np.allclose(torque, [0.015, 0.0, 0.0])
    assert updated[0] == pytest.approx(0.00019995896492062487)
    torque_sat, frozen = rate_torque(
        np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3),
        p, i, d, ff, np.array([True, False, False]), np.zeros(3, dtype=bool), limit, 0.01,
    )
    assert torque_sat[0] == pytest.approx(0.15)
    assert frozen[0] == pytest.approx(0.0)
    _, clamped = rate_torque(
        np.zeros(3), np.array([1.0, 0.0, 0.0]), np.zeros(3), np.zeros(3),
        p, i, d, ff, np.zeros(3, dtype=bool), np.zeros(3, dtype=bool), limit, 10.0,
    )
    assert clamped[0] == pytest.approx(0.30)
