import copy
import math

import numpy as np
import pytest
import torch

from rotorpy.controllers.quadrotor_control import BatchedSE3Control
from rotorpy.frames import FrameConvention
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import BatchedMultirotor, BatchedMultirotorParams


def _one_crazyflie(control_abstraction):
    device = torch.device("cpu")
    params = copy.deepcopy(quad_params)
    params["motor_noise_std"] = 0.0
    batch_params = BatchedMultirotorParams([params], 1, device)
    hover = math.sqrt(params["mass"] * 9.81 / (4.0 * params["k_eta"]))
    state = {
        "x": torch.zeros(1, 3, dtype=torch.double),
        "v": torch.zeros(1, 3, dtype=torch.double),
        "q": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.double),
        "w": torch.zeros(1, 3, dtype=torch.double),
        "wind": torch.zeros(1, 3, dtype=torch.double),
        "rotor_speeds": torch.full((1, 4), hover, dtype=torch.double),
    }
    vehicle = BatchedMultirotor(
        batch_params,
        1,
        state,
        device,
        control_abstraction=control_abstraction,
        aero=False,
        integrator="rk4",
    )
    return vehicle, state, hover, params


def test_frd_cmd_ctatt_holds_height_and_negates_rotor_y_once():
    vehicle, state, _hover, params = _one_crazyflie("cmd_ctatt")
    authored_y = float(quad_params["rotor_pos"]["r1"][1])
    vehicle.set_frame(FrameConvention.FRD_NED)
    vehicle.set_frame(FrameConvention.FRD_NED)
    control = {
        "cmd_thrust": torch.tensor([[params["mass"] * 9.81]], dtype=torch.double),
        "cmd_q": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.double),
    }
    for _ in range(100):
        state = vehicle.step(state, control, 0.01)
    assert state["x"][0, 2].item() == pytest.approx(0.0, abs=1e-2)
    assert np.allclose(state["q"][0].detach().cpu().numpy(), [0.0, 0.0, 0.0, 1.0], atol=1e-2)
    assert vehicle.params.rotor_geometry[0, 0, 1].item() == pytest.approx(-authored_y)


def test_frd_batched_se3_hover_acceleration_is_up():
    device = torch.device("cpu")
    params = copy.deepcopy(quad_params)
    batch_params = BatchedMultirotorParams([params], 1, device)
    controller = BatchedSE3Control(batch_params, 1, device)
    controller.set_frame(FrameConvention.FRD_NED)
    controller.set_frame(FrameConvention.FRD_NED)
    state = {
        "x": torch.zeros(1, 3, dtype=torch.double),
        "v": torch.zeros(1, 3, dtype=torch.double),
        "q": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.double),
        "w": torch.zeros(1, 3, dtype=torch.double),
    }
    flat = {
        "x": torch.zeros(1, 3, dtype=torch.double),
        "x_dot": torch.zeros(1, 3, dtype=torch.double),
        "x_ddot": torch.zeros(1, 3, dtype=torch.double),
        "yaw": torch.zeros(1, dtype=torch.double),
        "yaw_dot": torch.zeros(1, dtype=torch.double),
    }
    cmd = controller.update(0.0, state, flat)
    authored_y = float(quad_params["rotor_pos"]["r1"][1])
    assert cmd["cmd_acc"][0, 2].item() == pytest.approx(-9.81, rel=1e-3)
    assert cmd["cmd_thrust"][0, 0].item() == pytest.approx(params["mass"] * 9.81, rel=1e-3)
    assert np.allclose(cmd["cmd_q"][0].detach().cpu().numpy(), [0.0, 0.0, 0.0, 1.0], atol=1e-2)
    assert batch_params.rotor_geometry[0, 0, 1].item() == pytest.approx(-authored_y)


def test_frd_batched_cmd_vel_hover_speeds_match_hover():
    vehicle, state, hover, _params = _one_crazyflie("cmd_vel")
    vehicle.set_frame(FrameConvention.FRD_NED)
    speeds = vehicle.get_cmd_motor_speeds(
        state,
        {"cmd_v": torch.zeros(1, 3, dtype=torch.double)},
        [0],
    )
    assert np.allclose(speeds.detach().cpu().numpy(), np.full(4, hover), rtol=1e-2, atol=1.0)


def test_update_mass_keeps_the_active_frame_weight():
    vehicle, _state, _hover, _params = _one_crazyflie("cmd_ctatt")
    authored_y = float(quad_params["rotor_pos"]["r1"][1])
    vehicle.params.update_mass(0, 0.02)
    assert np.allclose(vehicle.params.weight[0].detach().cpu().numpy(), [0.0, 0.0, -0.02 * 9.81])
    vehicle.set_frame(FrameConvention.FRD_NED)
    vehicle.set_frame(FrameConvention.FRD_NED)
    vehicle.params.update_mass(0, 0.05)
    assert np.allclose(vehicle.params.weight[0].detach().cpu().numpy(), [0.0, 0.0, 0.05 * 9.81])
    assert vehicle.params.rotor_geometry[0, 0, 1].item() == pytest.approx(-authored_y)
    vehicle.set_frame(FrameConvention.FLU_ENU)
    vehicle.params.update_mass(0, 0.04)
    assert np.allclose(vehicle.params.weight[0].detach().cpu().numpy(), [0.0, 0.0, -0.04 * 9.81])
    assert vehicle.params.rotor_geometry[0, 0, 1].item() == pytest.approx(authored_y)


def test_frd_batched_cmd_acc_hover_speeds_match_hover():
    vehicle, state, hover, _params = _one_crazyflie("cmd_acc")
    vehicle.set_frame(FrameConvention.FRD_NED)
    speeds = vehicle.get_cmd_motor_speeds(
        state,
        {"cmd_acc": torch.tensor([[0.0, 0.0, -9.81]], dtype=torch.double)},
        [0],
    )
    assert np.allclose(speeds.detach().cpu().numpy(), np.full(4, hover), rtol=1e-2, atol=1.0)
