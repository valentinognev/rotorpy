import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.frames import (
    FrameConvention,
    attitude_enu_flu_to_ned_frd,
    attitude_ned_frd_to_enu_flu,
    enu_to_ned,
    flu_to_frd,
    frd_to_flu,
    ned_to_enu,
)
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import FrameInvariantRK45, Multirotor


def test_set_frame_twice_negates_rotor_y_once():
    params = copy.deepcopy(quad_params)
    params["Ixy"] = 0.2
    vehicle = Multirotor(params, aero=False)
    authored_y = quad_params["rotor_pos"]["r1"][1]
    vehicle.set_frame(FrameConvention.FRD_NED)
    vehicle.set_frame(FrameConvention.FRD_NED)
    assert vehicle.rotor_pos["r1"][1] == pytest.approx(-authored_y)
    assert vehicle.inertia[0, 1] == pytest.approx(-0.2)
    assert vehicle.c_Dy == quad_params["c_Dy"]
    assert np.allclose(vehicle.weight, [0, 0, 0.03 * 9.81])
    assert np.allclose(vehicle.thrust_axis, [0, 0, -1])
    vehicle.set_frame(FrameConvention.FLU_ENU)
    assert vehicle.rotor_pos["r1"][1] == pytest.approx(authored_y)
    assert np.allclose(vehicle.weight, [0, 0, -0.03 * 9.81])


def test_frd_hover_holds_identity_and_height():
    vehicle = Multirotor(quad_params, aero=False)
    vehicle.set_frame(FrameConvention.FRD_NED)
    hover = np.sqrt(0.03 * 9.81 / (4 * quad_params["k_eta"]))
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0.0, 0.0, 0.0, 1.0]),
             "w": np.zeros(3), "wind": np.zeros(3), "rotor_speeds": np.ones(4) * hover}
    for _ in range(100):
        state = vehicle.step(state, {"cmd_motor_speeds": np.ones(4) * hover}, 0.01)
    assert state["x"][2] == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(state["q"], [0, 0, 0, 1], atol=1e-3)


def test_frd_cmd_vel_hover_speeds_match_hover():
    vehicle = Multirotor(quad_params, aero=False, control_abstraction="cmd_vel")
    vehicle.set_frame(FrameConvention.FRD_NED)
    hover = np.sqrt(0.03 * 9.81 / (4 * quad_params["k_eta"]))
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0., 0., 0., 1.]),
             "w": np.zeros(3), "wind": np.zeros(3), "rotor_speeds": np.ones(4) * hover}
    speeds = vehicle.get_cmd_motor_speeds(state, {"cmd_v": np.zeros(3), "cmd_yaw": 0.0})
    assert np.allclose(speeds, np.ones(4) * hover, rtol=1e-2, atol=1.0)


def test_ground_side_follows_down():
    flu = Multirotor(quad_params, aero=False, enable_ground=True)
    frd = Multirotor(quad_params, aero=False, enable_ground=True)
    frd.set_frame(FrameConvention.FRD_NED)
    assert flu._on_ground({"x": np.array([0.0, 0.0, 1.0])}) is False
    assert frd._on_ground({"x": np.array([0.0, 0.0, -1.0])}) is False
    assert frd._on_ground({"x": np.array([0.0, 0.0, 0.01])}) is True


def _accepted_steps(vehicle, state, cmd_speeds, t_final):
    packed = vehicle._pack_state(state)

    def derivative(t, s, vehicle=vehicle):
        return vehicle._s_dot_fn(t, s, cmd_speeds)

    solver = FrameInvariantRK45(derivative, 0.0, packed, t_final)
    steps = []
    while solver.status == "running":
        solver.step()
        if solver.t_old is not None and solver.t != solver.t_old:
            steps.append(float(solver.t - solver.t_old))
    return np.array(steps), solver.y


def test_rk45_steps_match_for_mapped_states():
    flu = Multirotor(quad_params, aero=True, enable_ground=False)
    frd = Multirotor(quad_params, aero=True, enable_ground=False)
    frd.set_frame(FrameConvention.FRD_NED)
    state = {
        "x": np.array([1.0, -2.0, 0.5]),
        "v": np.array([0.4, -0.2, 0.1]),
        "q": Rotation.from_euler("xyz", [0.2, -0.3, 0.4]).as_quat(),
        "w": np.array([0.3, -0.5, 0.2]),
        "wind": np.array([1.5, -0.7, 0.4]),
        "rotor_speeds": np.array([400.0, 420.0, 380.0, 410.0]),
    }
    frd_state = {
        "x": enu_to_ned(state["x"]),
        "v": enu_to_ned(state["v"]),
        "q": attitude_enu_flu_to_ned_frd(state["q"]),
        "w": flu_to_frd(state["w"]),
        "wind": enu_to_ned(state["wind"]),
        "rotor_speeds": np.array(state["rotor_speeds"], dtype=float, copy=True),
    }
    cmd = np.array([450.0, 430.0, 440.0, 420.0])
    flu_steps, flu_y = _accepted_steps(flu, state, cmd, 0.3)
    frd_steps, frd_y = _accepted_steps(frd, frd_state, cmd, 0.3)
    assert flu_steps.size > 1
    np.testing.assert_allclose(flu_steps, frd_steps, rtol=0.0, atol=1e-12)
    flu_out = flu._unpack_state(flu_y)
    frd_out = frd._unpack_state(frd_y)
    np.testing.assert_allclose(ned_to_enu(frd_out["x"]), flu_out["x"], atol=1e-8)
    np.testing.assert_allclose(ned_to_enu(frd_out["v"]), flu_out["v"], atol=1e-8)
    flu_q = flu_out["q"] / np.linalg.norm(flu_out["q"])
    frd_q = attitude_ned_frd_to_enu_flu(frd_out["q"])
    if np.dot(frd_q, flu_q) < 0.0:
        frd_q = -frd_q
    np.testing.assert_allclose(frd_q, flu_q, atol=1e-8)
    np.testing.assert_allclose(frd_to_flu(frd_out["w"]), flu_out["w"], atol=1e-8)
    np.testing.assert_allclose(ned_to_enu(frd_out["wind"]), flu_out["wind"], atol=1e-8)
    np.testing.assert_allclose(frd_out["rotor_speeds"], flu_out["rotor_speeds"], atol=1e-8)
