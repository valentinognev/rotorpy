import numpy as np
import pytest
from rotorpy.controllers.px4_cascade import Px4Cascade
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.frames import FrameConvention
from rotorpy.vehicles.hummingbird_params import quad_params

def test_allocation_matches_se3_in_both_frames():
    cascade = Px4Cascade(quad_params)
    se3 = SE3Control(quad_params)
    assert np.allclose(cascade.f_to_TM, se3.f_to_TM)
    assert np.allclose(cascade.TM_to_f, se3.TM_to_f)
    cascade.set_frame(FrameConvention.FRD_NED)
    cascade.set_frame(FrameConvention.FRD_NED)
    se3.set_frame(FrameConvention.FRD_NED)
    assert np.allclose(cascade.f_to_TM, se3.f_to_TM)
    with pytest.raises(ValueError):
        cascade.set_frame("north_east_up")


def _state(x=None, v=None, q=None, w=None):
    return {
        "x": np.zeros(3) if x is None else np.array(x, dtype=float),
        "v": np.zeros(3) if v is None else np.array(v, dtype=float),
        "q": np.array([0.0, 0.0, 0.0, 1.0]) if q is None else np.array(q, dtype=float),
        "w": np.zeros(3) if w is None else np.array(w, dtype=float),
    }


def test_velocity_tier_hovers_and_holds_ned_yaw_when_yaw_is_omitted():
    from rotorpy.guidance.scene import hover_rotor_speed
    cascade = Px4Cascade(quad_params)
    cascade.set_frame(FrameConvention.FRD_NED)
    out = cascade.update(0.0, _state(), {"tier": "velocity", "velocity_ned": np.zeros(3), "yaw": 0.0})
    assert out["cmd_thrust"] == pytest.approx(0.5 * 9.81)
    assert np.allclose(out["cmd_moment"], 0.0, atol=1e-8)
    assert np.allclose(out["cmd_motor_speeds"], hover_rotor_speed(quad_params))
    for key in (
        "cmd_motor_speeds", "cmd_motor_thrusts", "cmd_thrust", "cmd_moment",
        "cmd_q", "cmd_w", "cmd_v", "cmd_acc",
    ):
        assert key in out
    limited = Px4Cascade(quad_params)
    limited.set_frame(FrameConvention.FRD_NED)
    limited_out = limited.update(
        0.0,
        _state(),
        {"tier": "velocity", "velocity_ned": np.array([10.0, 0.0, -10.0]), "yaw": 0.0},
    )
    assert limited_out["cmd_thrust"] == pytest.approx(9.81)
    flu = Px4Cascade(quad_params)
    held = flu.update(0.0, _state(), {"tier": "velocity", "velocity_ned": np.zeros(3)})
    assert np.allclose(held["cmd_motor_speeds"], hover_rotor_speed(quad_params), rtol=1e-6)
    fed = Px4Cascade(quad_params)
    fed.set_frame(FrameConvention.FRD_NED)
    fed_out = fed.update(
        0.0,
        _state(),
        {"tier": "velocity", "velocity_ned": np.zeros(3), "acceleration_ned": np.array([0.0, 0.0, 1.0]), "yaw": 0.0},
    )
    assert fed_out["cmd_acc"][2] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        fed.update(0.01, _state(), {"tier": "position"})


def test_first_sample_integrates_for_one_hundredth_second():
    cascade = Px4Cascade(quad_params)
    cascade.set_frame(FrameConvention.FRD_NED)
    command = {"tier": "velocity", "velocity_ned": np.array([0.0, 0.0, 0.1]), "yaw": 0.0}
    first = cascade.update(0.0, _state(), command)
    second = cascade.update(0.05, _state(), command)
    third = cascade.update(0.09, _state(), command)
    assert first["cmd_thrust"] == pytest.approx(4.705)
    assert second["cmd_thrust"] == pytest.approx(4.704)
    # Third thrust includes dt from the second call: 0.05 - 0.0, not 0.09 - 0.05.
    assert third["cmd_thrust"] == pytest.approx(4.699)


def test_velocity_d_term_is_a_five_hertz_low_pass_and_starts_at_zero():
    cascade = Px4Cascade(quad_params)
    cascade.set_frame(FrameConvention.FRD_NED)
    command = {"tier": "velocity", "velocity_ned": np.zeros(3), "yaw": 0.0}
    opening = cascade.update(0.0, _state(v=[0.01, 0.0, 0.0]), command)
    assert opening["cmd_acc"][0] == pytest.approx(-0.018)
    moving = Px4Cascade(quad_params)
    moving.set_frame(FrameConvention.FRD_NED)
    moving.update(0.0, _state(), command)
    stepped = moving.update(0.01, _state(v=[0.01, 0.0, 0.0]), command)
    alpha = 0.01 / (0.01 + 1.0 / (2.0 * np.pi * 5.0))
    assert stepped["cmd_acc"][0] == pytest.approx(-0.018 - 0.2 * alpha)


def test_position_tier_clamps_velocity_and_the_velocity_tier_does_not():
    def position(sp):
        controller = Px4Cascade(quad_params)
        controller.set_frame(FrameConvention.FRD_NED)
        return controller.update(0.0, _state(), {
            "tier": "position",
            "position_ned": np.array(sp, dtype=float),
            "yaw": 0.0,
        })

    def velocity(sp, measured=None):
        controller = Px4Cascade(quad_params)
        controller.set_frame(FrameConvention.FRD_NED)
        return controller.update(0.0, _state(v=measured), {
            "tier": "velocity",
            "velocity_ned": np.array(sp, dtype=float),
            "yaw": 0.0,
        })

    down = position([0.0, 0.0, 19.0])
    assert down["cmd_v"][2] == pytest.approx(1.0)
    assert down["cmd_thrust"] == pytest.approx(2.905)
    free_down = velocity([0.0, 0.0, 19.0])
    assert free_down["cmd_v"][2] == pytest.approx(19.0)
    assert free_down["cmd_thrust"] == pytest.approx(0.12 * 9.81)
    held = position([20.0, 0.0, 0.0])
    # Rebuild at the measured velocity the cap is compared against.
    capped = Px4Cascade(quad_params)
    capped.set_frame(FrameConvention.FRD_NED)
    capped_out = capped.update(0.0, _state(v=[11.0, 0.0, 0.0]), {
        "tier": "position",
        "position_ned": np.array([20.0, 0.0, 0.0]),
        "yaw": 0.0,
    })
    assert np.allclose(capped_out["cmd_v"], [12.0, 0.0, 0.0])
    assert capped_out["cmd_thrust"] == pytest.approx(4.986885300465612)
    fast = velocity([19.0, 0.0, 0.0], measured=[11.0, 0.0, 0.0])
    assert np.allclose(fast["cmd_v"], [19.0, 0.0, 0.0])
    assert fast["cmd_thrust"] == pytest.approx(6.936717523440031)
    assert held["cmd_v"][0] == pytest.approx(12.0)
    fed = Px4Cascade(quad_params)
    fed.set_frame(FrameConvention.FRD_NED)
    fed_out = fed.update(0.0, _state(), {
        "tier": "position",
        "position_ned": np.zeros(3),
        "acceleration_ned": np.array([0.0, 0.0, 1.0]),
        "yaw": 0.0,
    })
    assert fed_out["cmd_acc"][2] == pytest.approx(1.0)


def test_attitude_tier_holds_thrust_and_adds_yaw_rate():
    from rotorpy.guidance.scene import hover_rotor_speed
    identity = np.array([0.0, 0.0, 0.0, 1.0])
    cascade = Px4Cascade(quad_params)
    cascade.set_frame(FrameConvention.FRD_NED)
    held = cascade.update(0.0, _state(), {"tier": "attitude", "attitude": identity, "thrust": 0.5})
    assert held["cmd_thrust"] == pytest.approx(0.5 * 9.81)
    assert np.allclose(held["cmd_motor_speeds"], hover_rotor_speed(quad_params), rtol=1e-6)
    assert np.allclose(held["cmd_v"], 0.0)
    yawing = Px4Cascade(quad_params)
    yawing.set_frame(FrameConvention.FRD_NED)
    out = yawing.update(
        0.0, _state(), {"tier": "attitude", "attitude": identity, "thrust": 0.5, "yaw_rate": 0.3}
    )
    assert out["cmd_moment"][2] == pytest.approx(0.06)
    assert np.allclose(out["cmd_w"], [0.0, 0.0, 0.3])


def test_rate_tier_freezes_a_saturated_pitch_integral_in_the_flu_frame():
    # 0.8 N·m is past the Hummingbird pitch stop. 50 rad/s would also make
    # the 400° i-factor zero the integral, which hides a missing flag.
    huge = {"tier": "rate", "rate_frd": np.array([0.0, 0.8 / 0.15, 0.0]), "thrust": 0.5}
    cascade = Px4Cascade(quad_params)
    first = cascade.update(0.0, _state(), huge)
    second = cascade.update(0.01, _state(), huge)
    third = cascade.update(0.02, _state(), huge)
    assert np.any(np.isclose(first["cmd_motor_speeds"], 0.0))
    assert second["cmd_moment"][1] < first["cmd_moment"][1] - 1e-4
    assert third["cmd_moment"][1] == pytest.approx(second["cmd_moment"][1], abs=1e-6)
    small = {"tier": "rate", "rate_frd": np.array([0.0, 0.05, 0.0]), "thrust": 0.5}
    unsaturated = Px4Cascade(quad_params)
    opened = unsaturated.update(0.0, _state(), small)
    grown = unsaturated.update(0.01, _state(), small)
    kept = unsaturated.update(0.02, _state(), small)
    assert grown["cmd_moment"][1] < opened["cmd_moment"][1]
    assert kept["cmd_moment"][1] < grown["cmd_moment"][1]
    matched = Px4Cascade(quad_params)
    quiet = matched.update(
        0.0,
        _state(w=[0.0, 0.1, 0.0]),
        {"tier": "rate", "rate_frd": np.array([0.0, -0.1, 0.0]), "thrust": 0.5},
    )
    assert quiet["cmd_moment"][1] == pytest.approx(0.0, abs=1e-6)


def test_flat_output_is_the_position_tier_in_the_simulator_frame():
    from rotorpy.frames import enu_to_ned, yaw_enu_to_ned, yaw_rate_enu_to_ned
    flat = {
        "x": np.zeros(3),
        "x_dot": np.array([1.0, 2.0, 3.0]),
        "x_ddot": np.array([0.1, -0.2, 0.3]),
        "x_dddot": np.array([9.0, 9.0, 9.0]),
        "yaw": 0.4,
        "yaw_dot": -0.2,
    }
    flu = Px4Cascade(quad_params)
    explicit = Px4Cascade(quad_params)
    flu_out = flu.update(0.0, _state(), flat)
    explicit_out = explicit.update(0.0, _state(), {
        "tier": "position",
        "position_ned": enu_to_ned(flat["x"]),
        "velocity_ned": enu_to_ned(flat["x_dot"]),
        "acceleration_ned": enu_to_ned(flat["x_ddot"]),
        "yaw": yaw_enu_to_ned(flat["yaw"]),
        "yaw_rate": yaw_rate_enu_to_ned(flat["yaw_dot"]),
    })
    assert np.allclose(flu_out["cmd_motor_speeds"], explicit_out["cmd_motor_speeds"])
    assert np.allclose(flu_out["cmd_q"], explicit_out["cmd_q"])
    ned = Px4Cascade(quad_params)
    ned.set_frame(FrameConvention.FRD_NED)
    direct = Px4Cascade(quad_params)
    direct.set_frame(FrameConvention.FRD_NED)
    ned_out = ned.update(0.0, _state(), flat)
    direct_out = direct.update(0.0, _state(), {
        "tier": "position",
        "position_ned": flat["x"],
        "velocity_ned": flat["x_dot"],
        "acceleration_ned": flat["x_ddot"],
        "yaw": flat["yaw"],
        "yaw_rate": flat["yaw_dot"],
    })
    assert np.allclose(ned_out["cmd_v"], direct_out["cmd_v"])
    assert not np.allclose(ned_out["cmd_v"], flu_out["cmd_v"])
