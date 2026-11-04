"""Gazebo homing offboard command modes."""

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.controllers.px4_cascade.gains import GAINS
from rotorpy.controllers.px4_cascade.math import attitude_rate_setpoint
from rotorpy.guidance.homing_config import (
    command_mode_from_config,
    load_homing_config,
)
from rotorpy.guidance.offboard_command import (
    HoverThrustEstimator,
    OFFBOARD_PX4_PARAMETERS,
    AttitudeSetpoint,
    CommandMode,
    OffboardGains,
    attitude_setpoint,
    offboard_gains,
    parse_command_mode,
    rate_setpoint,
)

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"
CONFIG = {"track_tau_s": 0.5, "max_tilt_deg": 45.0, "min_specific_up_g": 0.3}
ZERO = [0.0, 0.0, 0.0]


def _gains(**overrides):
    gains, _ = offboard_gains(CONFIG, {})
    return dataclasses.replace(gains, **overrides)


def _euler(setpoint):
    yaw, pitch, roll = Rotation.from_quat(setpoint.quat_xyzw).as_euler("ZYX")
    return yaw, pitch, roll


def test_parse_command_mode_accepts_three_modes():
    assert parse_command_mode("attitude") is CommandMode.ATTITUDE
    assert parse_command_mode("rate") is CommandMode.RATE
    assert parse_command_mode("velocity") is CommandMode.VELOCITY


def test_parse_command_mode_rejects_unknown():
    with pytest.raises(ValueError, match="attitude"):
        parse_command_mode("angles")


def test_missing_key_defaults_to_attitude():
    assert command_mode_from_config({}) is CommandMode.ATTITUDE


def test_example_config_selects_attitude():
    config = load_homing_config(_CONFIG_PATH)
    assert command_mode_from_config(config) is CommandMode.ATTITUDE


def test_load_rejects_bad_mode(tmp_path):
    config = load_homing_config(_CONFIG_PATH)
    config["command_mode"] = "bogus"
    bad_path = tmp_path / "bad.jsonc"
    bad_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="attitude"):
        load_homing_config(bad_path)


def test_example_config_carries_the_tracking_keys():
    config = load_homing_config(_CONFIG_PATH)
    for key in CONFIG:
        assert config[key] == CONFIG[key]


def test_offboard_gains_prefer_px4_and_report_fallbacks():
    params = {"MPC_THR_HOVER": 0.62, "MC_ROLLRATE_MAX": 180.0, "MPC_TILTMAX_AIR": 30.0}
    gains, fallback = offboard_gains(CONFIG, params)
    assert gains.hover_thrust == 0.62
    assert gains.rate_limit_rad[0] == pytest.approx(np.deg2rad(180.0))
    assert gains.rate_limit_rad[1] == pytest.approx(np.deg2rad(GAINS["MC_PITCHRATE_MAX"]))
    assert gains.tilt_max_rad == pytest.approx(np.deg2rad(30.0))
    assert gains.roll_p == GAINS["MC_ROLL_P"]
    assert "MC_ROLL_P" in fallback
    assert "MPC_THR_HOVER" not in fallback
    assert "MC_ROLLRATE_MAX" not in fallback
    assert (gains.track_tau_s, gains.max_tilt_deg, gains.min_specific_up_g) == (0.5, 45.0, 0.3)


def test_offboard_gains_without_px4_fall_back_on_every_parameter():
    gains, fallback = offboard_gains(CONFIG, {})
    assert fallback == list(OFFBOARD_PX4_PARAMETERS)
    assert gains.hover_thrust == GAINS["MPC_THR_HOVER"]
    assert gains.thr_min == GAINS["MPC_THR_MIN"]
    assert gains.thr_max == GAINS["MPC_THR_MAX"]
    assert gains.thr_xy_margin == GAINS["MPC_THR_XY_MARG"]
    assert gains.yaw_weight == GAINS["MC_YAW_WEIGHT"]
    assert gains.rate_limit_rad == pytest.approx(
        np.deg2rad([GAINS["MC_ROLLRATE_MAX"], GAINS["MC_PITCHRATE_MAX"], GAINS["MC_YAWRATE_MAX"]])
    )


def test_offboard_gains_treat_non_finite_px4_values_as_missing():
    gains, fallback = offboard_gains(CONFIG, {"MPC_THR_HOVER": float("nan")})
    assert gains.hover_thrust == GAINS["MPC_THR_HOVER"]
    assert "MPC_THR_HOVER" in fallback


def test_zero_error_gives_configured_hover_thrust():
    sp = attitude_setpoint(ZERO, ZERO, 0.3, ZERO, _gains(hover_thrust=0.6))
    assert isinstance(sp, AttitudeSetpoint)
    assert sp.body_rates_frd is None
    assert sp.thrust == pytest.approx(0.6)
    assert _euler(sp) == pytest.approx((0.3, 0.0, 0.0), abs=1e-9)


def test_north_velocity_demand_pitches_nose_down():
    sp = attitude_setpoint([5, 0, 0], ZERO, 0.0, ZERO, _gains())
    _, pitch, roll = _euler(sp)
    assert pitch < -0.1
    assert abs(roll) < 1e-9
    assert sp.thrust > _gains().hover_thrust


def test_east_velocity_demand_rolls_right():
    _, pitch, roll = _euler(attitude_setpoint([0, 5, 0], ZERO, 0.0, ZERO, _gains()))
    assert roll > 0.1
    assert abs(pitch) < 1e-9


def test_yaw_does_not_rotate_the_demand_frame():
    # Yaw only orients the nose; a north demand tilts the thrust south at any yaw.
    sp = attitude_setpoint([5, 0, 0], ZERO, np.pi / 2, ZERO, _gains())
    body_z = Rotation.from_quat(sp.quat_xyzw).apply([0, 0, 1])
    assert body_z[0] < -0.1
    assert abs(body_z[1]) < 1e-9


def test_climb_demand_raises_thrust_and_descent_lowers_it():
    hover = _gains().hover_thrust
    assert attitude_setpoint([0, 0, -3], ZERO, 0, ZERO, _gains()).thrust > hover
    assert attitude_setpoint([0, 0, 3], ZERO, 0, ZERO, _gains()).thrust < hover


def test_measured_velocity_cancels_the_demand():
    sp = attitude_setpoint([5, 0, -1], ZERO, 0.0, [5, 0, -1], _gains())
    assert sp.thrust == pytest.approx(_gains().hover_thrust)
    assert _euler(sp)[1:] == pytest.approx((0.0, 0.0), abs=1e-9)


def test_tilt_never_exceeds_max_tilt():
    sp = attitude_setpoint([100, 100, 0], ZERO, 0, ZERO, _gains())
    body_z = Rotation.from_quat(sp.quat_xyzw).apply([0, 0, 1])
    assert np.degrees(np.arccos(body_z[2])) <= 45.0 + 1e-6


def test_thrust_stays_inside_px4_limits():
    gains = _gains()
    for demand in ([0, 0, -50], [0, 0, 50], [100, 100, -50]):
        thrust = attitude_setpoint(demand, ZERO, 0, ZERO, gains).thrust
        assert gains.thr_min <= thrust <= gains.thr_max


def test_feedforward_ignored_unless_enabled():
    gains = _gains()
    quiet = attitude_setpoint(ZERO, ZERO, 0, ZERO, gains)
    off = attitude_setpoint(ZERO, [3, 0, 0], 0, ZERO, gains)
    on = attitude_setpoint(ZERO, [3, 0, 0], 0, ZERO, gains, enable_acceleration_feedforward=True)
    assert np.allclose(off.quat_xyzw, quiet.quat_xyzw)
    assert off.thrust == pytest.approx(quiet.thrust)
    assert _euler(on)[1] < -0.1


def test_quaternion_is_unit_with_non_negative_w():
    for yaw in np.linspace(-np.pi, np.pi, 13):
        sp = attitude_setpoint([3, -2, -1], ZERO, yaw, ZERO, _gains())
        assert np.linalg.norm(sp.quat_xyzw) == pytest.approx(1.0)
        assert sp.quat_xyzw[3] >= 0.0


@pytest.mark.parametrize("field", ["velocity", "acceleration", "yaw", "measured"])
def test_non_finite_input_raises(field):
    args = {"velocity": ZERO, "acceleration": ZERO, "yaw": 0.0, "measured": ZERO}
    args[field] = float("nan") if field == "yaw" else [np.inf, 0.0, 0.0]
    with pytest.raises(ValueError):
        attitude_setpoint(args["velocity"], args["acceleration"], args["yaw"], args["measured"], _gains())


LEVEL = [0.0, 0.0, 0.0, 1.0]


def test_rates_zero_when_at_setpoint():
    sp = attitude_setpoint(ZERO, ZERO, 0.4, ZERO, _gains())
    out = rate_setpoint(sp, sp.quat_xyzw, _gains())
    assert out.body_rates_frd == pytest.approx(np.zeros(3), abs=1e-9)


def test_rates_point_toward_setpoint_and_keep_thrust():
    gains = _gains()
    nose_down = attitude_setpoint([5, 0, 0], ZERO, 0.0, ZERO, gains)
    out = rate_setpoint(nose_down, LEVEL, gains)
    assert out.body_rates_frd[1] < -0.1
    assert abs(out.body_rates_frd[0]) < 1e-9
    assert out.thrust == nose_down.thrust
    assert out.quat_xyzw is nose_down.quat_xyzw
    assert nose_down.body_rates_frd is None

    right_roll = attitude_setpoint([0, 5, 0], ZERO, 0.0, ZERO, gains)
    assert rate_setpoint(right_roll, LEVEL, gains).body_rates_frd[0] > 0.1

    yaw_right = attitude_setpoint(ZERO, ZERO, 0.5, ZERO, gains)
    assert rate_setpoint(yaw_right, LEVEL, gains).body_rates_frd[2] > 0.1


def test_rates_use_the_gains_and_a_zero_yaw_rate():
    gains = _gains()
    sp = attitude_setpoint([2, 0, 0], ZERO, 0.0, ZERO, gains)
    expected = attitude_rate_setpoint(
        np.array(LEVEL), sp.quat_xyzw, 0.0, gains.roll_p, gains.pitch_p,
        gains.yaw_p, gains.yaw_weight, gains.rate_limit_rad,
    )
    assert rate_setpoint(sp, LEVEL, gains).body_rates_frd == pytest.approx(expected)


def test_rates_are_clamped_for_large_error():
    sp = attitude_setpoint(ZERO, ZERO, np.pi, ZERO, _gains())
    current = Rotation.from_euler("ZYX", [0, 0, np.radians(80)]).as_quat()
    gains = _gains()
    rates = rate_setpoint(sp, current, gains).body_rates_frd
    assert np.all(np.isfinite(rates))
    assert np.all(np.abs(rates) <= gains.rate_limit_rad + 1e-9)

    tight = _gains(rate_limit_rad=np.array([0.1, 0.2, 0.3]))
    big = attitude_setpoint([100, 100, 0], ZERO, np.pi, ZERO, tight)
    clamped = rate_setpoint(big, LEVEL, tight).body_rates_frd
    assert np.abs(clamped) == pytest.approx(tight.rate_limit_rad)


def test_rates_accept_a_non_unit_current_quaternion():
    sp = attitude_setpoint([5, 0, 0], ZERO, 0.0, ZERO, _gains())
    unit = rate_setpoint(sp, LEVEL, _gains()).body_rates_frd
    scaled = rate_setpoint(sp, [0.0, 0.0, 0.0, 2.0], _gains()).body_rates_frd
    assert scaled == pytest.approx(unit)


@pytest.mark.parametrize(
    "current",
    [[0, 0, 0, 0], [np.nan, 0, 0, 1], [np.inf, 0, 0, 1], [0, 0, 1]],
)
def test_bad_current_attitude_raises(current):
    sp = attitude_setpoint(ZERO, ZERO, 0.0, ZERO, _gains())
    with pytest.raises(ValueError):
        rate_setpoint(sp, current, _gains())


def test_descent_acceleration_is_limited_to_px4_acc_down_max():
    gains = _gains()
    expected = gains.hover_thrust * (1.0 - gains.acc_down_max / 9.80665)
    for demand in (8.0, 20.0):
        sp = attitude_setpoint([0, 0, demand], ZERO, 0, ZERO, gains)
        assert sp.thrust == pytest.approx(expected, rel=1e-2)


def test_climb_acceleration_is_limited_to_px4_acc_up_max():
    gains = _gains()
    expected = gains.hover_thrust * (1.0 + gains.acc_up_max / 9.80665)
    for demand in (-8.0, -20.0):
        sp = attitude_setpoint([0, 0, demand], ZERO, 0, ZERO, gains)
        assert sp.thrust == pytest.approx(expected, rel=1e-2)


def test_horizontal_acceleration_is_limited_to_px4_acc_hor():
    gains = _gains()
    expected_tilt = np.arctan(gains.acc_hor / 9.80665)
    sp = attitude_setpoint([30.0, 40.0, 0], ZERO, 0, ZERO, gains)
    _, pitch, roll = _euler(sp)
    assert np.hypot(pitch, roll) == pytest.approx(expected_tilt, rel=2e-2)


def test_offboard_gains_read_px4_acceleration_limits():
    gains, fallback = offboard_gains(CONFIG, {"MPC_ACC_DOWN_MAX": 2.0, "MPC_ACC_HOR": 7.0})
    assert (gains.acc_down_max, gains.acc_hor) == (2.0, 7.0)
    assert gains.acc_up_max == 4.0
    assert "MPC_ACC_UP_MAX" in fallback


def test_hover_thrust_estimator_is_none_without_enough_samples():
    estimator = HoverThrustEstimator(min_samples=5)
    for _ in range(4):
        estimator.add(0.55)
    assert estimator.estimate() is None


def test_hover_thrust_estimator_returns_the_median_of_recent_samples():
    estimator = HoverThrustEstimator(min_samples=3, max_samples=5)
    for value in (0.9, 0.9, 0.55, 0.56, 0.54, 0.55, 0.57):
        estimator.add(value)
    assert estimator.estimate() == pytest.approx(0.55)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, 0.05, 0.95, 1.5, -0.3])
def test_hover_thrust_estimator_ignores_implausible_samples(bad):
    estimator = HoverThrustEstimator(min_samples=2)
    estimator.add(bad)
    estimator.add(0.5)
    estimator.add(0.5)
    assert estimator.estimate() == pytest.approx(0.5)
    assert len(estimator) == 2
