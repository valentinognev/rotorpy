"""`command_limits.limit_command`: one coherent vehicle-envelope/continuity limiter
for a guidance command (plan Phase 7). Property-style tests with random and explicit
non-finite inputs, plus the visibility-margin speed scale it composes.
"""

import numpy as np
import pytest

from rotorpy.guidance.command_limits import LimitedCommand, VehicleEnvelope, limit_command, visibility_speed_scale

ENVELOPE = VehicleEnvelope(
    max_speed_m_s=15.0,
    max_accel_m_s2=5.0,
    max_jerk_m_s3=10.0,
    max_yaw_rate_rad_s=1.0,
    max_vertical_speed_m_s=3.0,
    visibility_margin_frac=0.2,
    visibility_min_speed_scale=0.3,
)


def _rng():
    return np.random.default_rng(12345)


def test_within_envelope_request_passes_through_unchanged_with_no_reasons():
    velocity = np.array([1.0, 0.5, 0.2])
    acceleration = np.array([0.1, 0.0, 0.0])
    result = limit_command(
        velocity, acceleration, 0.3,
        previous_velocity_ned=velocity, previous_acceleration_ned=acceleration, previous_yaw_rad=0.3,
        dt=0.05, envelope=ENVELOPE,
    )
    assert isinstance(result, LimitedCommand)
    assert np.allclose(result.velocity_ned, velocity)
    assert np.allclose(result.acceleration_ned, acceleration)
    assert result.yaw_rad == pytest.approx(0.3)
    assert result.reasons == []


def test_oversized_speed_is_clamped_to_the_total_speed_envelope_and_reasoned():
    # No continuity bound in play (dt=0): isolates the speed envelope itself.
    velocity = np.array([100.0, 0.0, 0.0])
    result = limit_command(
        velocity, np.zeros(3), 0.0,
        previous_velocity_ned=None, previous_acceleration_ned=None, previous_yaw_rad=None,
        dt=0.0, envelope=ENVELOPE,
    )
    assert np.isclose(np.linalg.norm(result.velocity_ned), ENVELOPE.max_speed_m_s)
    assert "speed_envelope" in result.reasons


def test_oversized_vertical_rate_is_clamped_and_reasoned():
    # Within the (generous) total-speed cap, so only the vertical-rate bound fires.
    velocity = np.array([0.0, 0.0, 10.0])
    result = limit_command(
        velocity, np.zeros(3), 0.0,
        previous_velocity_ned=None, previous_acceleration_ned=None, previous_yaw_rad=None,
        dt=0.0, envelope=ENVELOPE,
    )
    assert abs(result.velocity_ned[2]) <= ENVELOPE.max_vertical_speed_m_s + 1e-9
    assert "vertical_rate_envelope" in result.reasons


def test_large_velocity_step_is_bounded_by_max_accel_times_dt():
    previous = np.array([0.0, 0.0, 0.0])
    requested = np.array([10.0, 0.0, 0.0])
    dt = 0.05
    result = limit_command(
        requested, np.zeros(3), 0.0,
        previous_velocity_ned=previous, previous_acceleration_ned=np.zeros(3), previous_yaw_rad=0.0,
        dt=dt, envelope=ENVELOPE,
    )
    step = np.linalg.norm(result.velocity_ned - previous)
    assert step <= ENVELOPE.max_accel_m_s2 * dt + 1e-9
    assert "accel_continuity" in result.reasons


def test_large_acceleration_step_is_bounded_by_max_jerk_times_dt():
    previous_accel = np.array([0.0, 0.0, 0.0])
    requested_accel = np.array([5.0, 0.0, 0.0])
    dt = 0.05
    result = limit_command(
        np.zeros(3), requested_accel, 0.0,
        previous_velocity_ned=np.zeros(3), previous_acceleration_ned=previous_accel, previous_yaw_rad=0.0,
        dt=dt, envelope=ENVELOPE,
    )
    jerk_step = np.linalg.norm(result.acceleration_ned - previous_accel)
    assert jerk_step <= ENVELOPE.max_jerk_m_s3 * dt + 1e-9
    assert "jerk_envelope" in result.reasons


def test_large_yaw_step_is_bounded_by_max_yaw_rate_times_dt():
    dt = 0.05
    result = limit_command(
        np.zeros(3), np.zeros(3), np.pi,
        previous_velocity_ned=np.zeros(3), previous_acceleration_ned=np.zeros(3), previous_yaw_rad=0.0,
        dt=dt, envelope=ENVELOPE,
    )
    assert abs(result.yaw_rad) <= ENVELOPE.max_yaw_rate_rad_s * dt + 1e-9
    assert "yaw_rate_envelope" in result.reasons


def test_first_tick_with_no_previous_skips_continuity_but_still_applies_envelope():
    result = limit_command(
        np.array([100.0, 0.0, 0.0]), np.zeros(3), 0.0,
        previous_velocity_ned=None, previous_acceleration_ned=None, previous_yaw_rad=None,
        dt=0.0, envelope=ENVELOPE,
    )
    assert np.isclose(np.linalg.norm(result.velocity_ned), ENVELOPE.max_speed_m_s)
    assert "accel_continuity" not in result.reasons


@pytest.mark.parametrize("bad_velocity", [
    np.array([np.nan, 0.0, 0.0]),
    np.array([np.inf, 0.0, 0.0]),
    np.array([-np.inf, 1.0, 0.0]),
])
def test_non_finite_velocity_request_falls_back_to_the_previous_command(bad_velocity):
    previous = np.array([1.0, 2.0, 0.0])
    result = limit_command(
        bad_velocity, np.zeros(3), 0.0,
        previous_velocity_ned=previous, previous_acceleration_ned=np.zeros(3), previous_yaw_rad=0.0,
        dt=0.05, envelope=ENVELOPE,
    )
    assert np.allclose(result.velocity_ned, previous)
    assert result.reasons == ["non_finite_input_rejected"]


def test_non_finite_dt_falls_back_to_the_previous_command():
    previous = np.array([1.0, 2.0, 0.0])
    result = limit_command(
        np.array([1.0, 2.0, 0.0]), np.zeros(3), 0.0,
        previous_velocity_ned=previous, previous_acceleration_ned=np.zeros(3), previous_yaw_rad=0.0,
        dt=np.inf, envelope=ENVELOPE,
    )
    assert np.allclose(result.velocity_ned, previous)
    assert result.reasons == ["non_finite_input_rejected"]


def test_non_finite_request_with_no_previous_falls_back_to_zero():
    result = limit_command(
        np.array([np.nan, 0.0, 0.0]), np.zeros(3), 0.0,
        previous_velocity_ned=None, previous_acceleration_ned=None, previous_yaw_rad=None,
        dt=0.0, envelope=ENVELOPE,
    )
    assert np.allclose(result.velocity_ned, 0.0)
    assert np.isfinite(result.yaw_rad)
    assert result.reasons == ["non_finite_input_rejected"]


def test_visibility_margin_scales_speed_down_near_the_image_edge():
    scale_centered, reason_centered = visibility_speed_scale((160.0, 120.0), 320, 240, ENVELOPE)
    scale_edge, reason_edge = visibility_speed_scale((2.0, 120.0), 320, 240, ENVELOPE)
    assert scale_centered == 1.0 and reason_centered is None
    assert 0.0 < scale_edge < 1.0
    assert reason_edge == "visibility_margin"


def test_visibility_margin_feeds_into_limit_command_speed_cap():
    velocity = np.array([ENVELOPE.max_speed_m_s, 0.0, 0.0])
    result = limit_command(
        velocity, np.zeros(3), 0.0,
        previous_velocity_ned=velocity, previous_acceleration_ned=np.zeros(3), previous_yaw_rad=0.0,
        dt=0.05, envelope=ENVELOPE,
        center_px=(2.0, 120.0), image_width_px=320, image_height_px=240,
    )
    assert np.linalg.norm(result.velocity_ned) < ENVELOPE.max_speed_m_s
    assert "visibility_margin" in result.reasons
    assert "speed_envelope" in result.reasons


def test_visibility_margin_also_scales_down_the_yaw_rate_cap_not_only_speed():
    """"speed/turn limiting when the target nears the image edge": a hard turn must
    not itself be what loses the target off-frame."""
    dt = 0.05
    args = dict(
        previous_velocity_ned=np.zeros(3), previous_acceleration_ned=np.zeros(3), previous_yaw_rad=0.0,
        dt=dt, envelope=ENVELOPE,
    )
    centered = limit_command(np.zeros(3), np.zeros(3), np.pi, **args, center_px=(160.0, 120.0),
                              image_width_px=320, image_height_px=240)
    near_edge = limit_command(np.zeros(3), np.zeros(3), np.pi, **args, center_px=(2.0, 120.0),
                               image_width_px=320, image_height_px=240)
    assert abs(near_edge.yaw_rad) < abs(centered.yaw_rad)
    assert "yaw_rate_envelope" in centered.reasons and "yaw_rate_envelope" in near_edge.reasons
    assert "visibility_margin" in near_edge.reasons


def test_property_random_requests_stay_finite_and_within_envelope_and_continuity():
    rng = _rng()
    previous_velocity = np.zeros(3)
    previous_acceleration = np.zeros(3)
    previous_yaw = 0.0
    dt = 0.05
    for _ in range(200):
        requested_velocity = rng.uniform(-200.0, 200.0, size=3)
        requested_acceleration = rng.uniform(-200.0, 200.0, size=3)
        requested_yaw = rng.uniform(-100.0, 100.0)
        result = limit_command(
            requested_velocity, requested_acceleration, requested_yaw,
            previous_velocity_ned=previous_velocity, previous_acceleration_ned=previous_acceleration,
            previous_yaw_rad=previous_yaw, dt=dt, envelope=ENVELOPE,
        )
        assert np.all(np.isfinite(result.velocity_ned))
        assert np.all(np.isfinite(result.acceleration_ned))
        assert np.isfinite(result.yaw_rad)
        assert np.linalg.norm(result.velocity_ned) <= ENVELOPE.max_speed_m_s + 1e-6
        assert abs(result.velocity_ned[2]) <= ENVELOPE.max_vertical_speed_m_s + 1e-6
        assert np.linalg.norm(result.acceleration_ned) <= ENVELOPE.max_accel_m_s2 + 1e-6
        step = np.linalg.norm(result.velocity_ned - previous_velocity)
        assert step <= ENVELOPE.max_accel_m_s2 * dt + 1e-6
        jerk_step = np.linalg.norm(result.acceleration_ned - previous_acceleration)
        assert jerk_step <= ENVELOPE.max_jerk_m_s3 * dt + 1e-6
        yaw_delta = abs(np.arctan2(np.sin(result.yaw_rad - previous_yaw), np.cos(result.yaw_rad - previous_yaw)))
        assert yaw_delta <= ENVELOPE.max_yaw_rate_rad_s * dt + 1e-6
        previous_velocity, previous_acceleration, previous_yaw = (
            result.velocity_ned, result.acceleration_ned, result.yaw_rad
        )
