"""Unit tests for the offline reliability harness's truth-side vehicle-response
model (plan Phase 9): first-order velocity tracking with accel/tilt limits,
command latency, and a delayed-pose history for the synthetic camera.
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.guidance.validation.vehicle_model import VehicleResponseModel, attitude_from_specific_force


def _model(**overrides):
    kwargs = dict(
        position_ned=np.zeros(3),
        velocity_ned=np.zeros(3),
        yaw_rad=0.0,
        tau_s=0.5,
        max_accel_m_s2=5.0,
        max_tilt_deg=45.0,
        min_specific_up_g=0.3,
        max_yaw_rate_rad_s=1.0,
        command_latency_s=0.0,
        image_delay_s=0.0,
    )
    kwargs.update(overrides)
    return VehicleResponseModel(**kwargs)


def test_velocity_converges_toward_the_commanded_goal():
    model = _model()
    goal = np.array([10.0, 0.0, 0.0])
    model.enqueue_command(0.0, goal, 0.0)
    t = 0.0
    for _ in range(2000):
        model.step(0.01, t + 0.01)
        t += 0.01
    assert np.linalg.norm(model.velocity_ned - goal) < 0.05


def test_acceleration_never_exceeds_the_configured_cap():
    model = _model(max_accel_m_s2=2.0, tau_s=0.05)
    model.enqueue_command(0.0, np.array([50.0, 0.0, 0.0]), 0.0)
    dt = 0.01
    t = 0.0
    previous_velocity = model.velocity_ned.copy()
    for _ in range(50):
        model.step(dt, t + dt)
        t += dt
        accel = (model.velocity_ned - previous_velocity) / dt
        assert np.linalg.norm(accel) <= 2.0 + 1e-6
        previous_velocity = model.velocity_ned.copy()


def test_attitude_tilt_from_vertical_stays_within_the_configured_limit():
    model = _model(max_tilt_deg=20.0, tau_s=0.05, max_accel_m_s2=50.0)
    model.enqueue_command(0.0, np.array([0.0, 30.0, 0.0]), 0.0)
    dt = 0.01
    t = 0.0
    max_tilt_seen = 0.0
    for _ in range(100):
        model.step(dt, t + dt)
        t += dt
        matrix = Rotation.from_quat(model.attitude_sim_quat).as_matrix()
        body_z_world = matrix[:, 2]
        tilt_rad = np.arccos(np.clip(body_z_world[2], -1.0, 1.0))
        max_tilt_seen = max(max_tilt_seen, tilt_rad)
    assert np.degrees(max_tilt_seen) <= 20.0 + 1e-6


def test_yaw_tracks_the_goal_within_the_rate_limit():
    model = _model(max_yaw_rate_rad_s=0.5)
    model.enqueue_command(0.0, np.zeros(3), np.pi)
    dt = 0.02
    t = 0.0
    previous_yaw = model.yaw_rad
    for _ in range(10):
        model.step(dt, t + dt)
        t += dt
        step = abs(model.yaw_rad - previous_yaw)
        assert step <= 0.5 * dt + 1e-9
        previous_yaw = model.yaw_rad


def test_command_latency_delays_when_a_new_goal_takes_effect():
    model = _model(command_latency_s=0.2, tau_s=0.2, max_accel_m_s2=50.0)
    model.enqueue_command(0.0, np.array([10.0, 0.0, 0.0]), 0.0)
    dt = 0.02
    t = 0.0
    for _ in range(5):
        model.step(dt, t + dt)
        t += dt
    # Still within the 0.2 s latency window: the goal has not taken effect yet.
    assert np.linalg.norm(model.velocity_ned) < 0.5
    for _ in range(300):
        model.step(dt, t + dt)
        t += dt
    assert np.linalg.norm(model.velocity_ned - np.array([10.0, 0.0, 0.0])) < 0.5


def test_delayed_pose_returns_an_older_sample_than_the_current_state():
    model = _model(image_delay_s=0.1)
    model.enqueue_command(0.0, np.array([5.0, 0.0, 0.0]), 0.0)
    dt = 0.01
    t = 0.0
    for _ in range(50):
        model.step(dt, t + dt)
        t += dt
    delayed = model.delayed_pose(t)
    current = model.pose_at(t)
    assert delayed.t_s < current.t_s
    assert np.linalg.norm(delayed.position_ned - model.position_ned) > 0.0


def test_non_finite_commands_never_corrupt_the_truth_state():
    model = _model()
    model.enqueue_command(0.0, np.array([np.nan, 0.0, 0.0]), 0.0)
    model.enqueue_command(0.0, np.array([1.0, 0.0, 0.0]), np.inf)
    model.step(0.02, 0.02)
    assert np.all(np.isfinite(model.position_ned))
    assert np.all(np.isfinite(model.velocity_ned))
    assert np.all(np.isfinite(model.attitude_sim_quat))
    assert np.isfinite(model.yaw_rad)


def test_attitude_from_specific_force_is_level_at_zero_acceleration():
    quat = attitude_from_specific_force(np.zeros(3), 0.0)
    expected = Rotation.from_euler("z", 0.0).as_quat()
    assert np.allclose(quat, expected, atol=1e-9) or np.allclose(quat, -expected, atol=1e-9)
