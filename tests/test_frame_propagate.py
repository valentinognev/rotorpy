import copy

import numpy as np
import pytest
import torch

from rotorpy.controllers.quadrotor_control import BatchedSE3Control, SE3Control
from rotorpy.environments import Environment
from rotorpy.estimators.nullestimator import NullEstimator
from rotorpy.estimators.wind_ukf import WindUKF
from rotorpy.frames import FrameConvention
from rotorpy.sensors.external_mocap import MotionCapture
from rotorpy.sensors.imu import BatchedImu, Imu
from rotorpy.simulate import simulate, simulate_batch
from rotorpy.trajectories.hover_traj import BatchedHoverTraj, HoverTraj
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.vehicles.multirotor import BatchedMultirotor, BatchedMultirotorParams, Multirotor
from rotorpy.wind.default_winds import BatchedNoWind, NoWind
from rotorpy.world import World

_EXTENTS = [-3, 3, -3, 3, -3, 3]
_MASS = 0.03
_G = 9.81


def hover_state():
    return {
        "x": np.zeros(3),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
    }


def hover_flat():
    return {
        "x": np.zeros(3),
        "x_dot": np.zeros(3),
        "x_ddot": np.zeros(3),
        "yaw": 0.0,
        "yaw_dot": 0.0,
    }


def _imu():
    return Imu(p_BS=np.zeros(3), R_BS=np.eye(3), sampling_rate=100)


def _mocap():
    return MotionCapture(sampling_rate=100, with_artifacts=False)


class _FramelessEstimator(NullEstimator):
    """Estimator simulate() can step, with no set_frame."""

    def __init__(self):
        super(_FramelessEstimator, self).__init__()
        self.marker = "untouched"


def _run_simulate(world, vehicle, controller, imu, estimator):
    return simulate(
        world,
        vehicle.initial_state,
        vehicle,
        controller,
        HoverTraj(),
        NoWind(),
        imu,
        _mocap(),
        estimator,
        0.01,
        0.01,
        0.25,
        False,
        terminate=False,
    )


def _two_drone_hover():
    num_drones = 2
    device = torch.device("cpu")
    params = [copy.deepcopy(quad_params), copy.deepcopy(quad_params)]
    for item in params:
        item["motor_noise_std"] = 0.0
    hover = float(np.sqrt(params[0]["mass"] * _G / (4.0 * params[0]["k_eta"])))
    initial_states = {
        "x": torch.zeros(num_drones, 3, dtype=torch.double),
        "v": torch.zeros(num_drones, 3, dtype=torch.double),
        "q": torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.double).repeat(num_drones, 1),
        "w": torch.zeros(num_drones, 3, dtype=torch.double),
        "wind": torch.zeros(num_drones, 3, dtype=torch.double),
        "rotor_speeds": torch.full((num_drones, 4), hover, dtype=torch.double),
    }
    batch_params = BatchedMultirotorParams(params, num_drones, device)
    vehicle = BatchedMultirotor(
        batch_params,
        num_drones,
        initial_states,
        device,
        control_abstraction="cmd_ctatt",
        aero=False,
        integrator="rk4",
    )
    controller = BatchedSE3Control(batch_params, num_drones, device)
    trajectory = BatchedHoverTraj(num_drones, device=device)
    wind = BatchedNoWind(num_drones)
    imu = BatchedImu(num_drones, device=device, sampling_rate=100)
    return initial_states, vehicle, controller, trajectory, wind, imu


class _UpdateCounter(object):
    def __init__(self, inner):
        self.inner = inner
        self.updates = 0

    def set_frame(self, frame):
        self.inner.set_frame(frame)

    def update(self, *args, **kwargs):
        self.updates += 1
        return self.inner.update(*args, **kwargs)


def _frd_hover_accel_is_level(estimator):
    """FRD hover thrust cancels gravity. FLU with the same input does not."""
    estimator.uk[0] = -_G
    x_next = estimator.f(np.zeros(9), 0.01)
    assert x_next[5] == pytest.approx(0.0, abs=1e-6)


def test_environment_pushes_frd_onto_vehicle_and_controller():
    world = World.empty([-3, 3, -3, 3, -3, 3], frame=FrameConvention.FRD_NED)
    env = Environment(Multirotor(quad_params, aero=False), SE3Control(quad_params),
                      HoverTraj(), world=world, sim_rate=100)
    assert np.allclose(env.vehicle.weight, [0, 0, 0.03 * 9.81])
    cmd = env.controller.update(0.0, hover_state(), hover_flat())
    assert np.allclose(cmd["cmd_q"], [0, 0, 0, 1], atol=1e-6)


def test_environment_pushes_frd_onto_imu_and_estimator():
    world = World.empty(_EXTENTS, frame=FrameConvention.FRD_NED)
    imu = _imu()
    imu.set_frame(FrameConvention.FLU_ENU)
    estimator = WindUKF(copy.deepcopy(quad_params), dt=0.01)
    estimator.set_frame(FrameConvention.FLU_ENU)
    Environment(
        Multirotor(copy.deepcopy(quad_params), aero=False),
        SE3Control(copy.deepcopy(quad_params)),
        HoverTraj(),
        imu=imu,
        estimator=estimator,
        world=world,
        sim_rate=100,
    )
    state = hover_state()
    statedot = {"vdot": np.zeros(3), "wdot": np.zeros(3)}
    measured = imu.measurement(state, statedot, with_noise=False)
    assert np.allclose(measured["accel"], [0.0, 0.0, -9.81], atol=1e-6)
    _frd_hover_accel_is_level(estimator)


def test_environment_skips_estimator_without_set_frame():
    world = World.empty(_EXTENTS, frame=FrameConvention.FRD_NED)
    estimator = _FramelessEstimator()
    env = Environment(
        Multirotor(copy.deepcopy(quad_params), aero=False),
        SE3Control(copy.deepcopy(quad_params)),
        HoverTraj(),
        estimator=estimator,
        world=world,
        sim_rate=100,
    )
    assert estimator.marker == "untouched"
    assert not hasattr(estimator, "set_frame")
    assert np.allclose(env.vehicle.weight, [0.0, 0.0, _MASS * _G])


def test_simulate_pushes_world_frame_onto_vehicle_imu_and_estimator():
    world = World.empty(_EXTENTS, frame=FrameConvention.FRD_NED)
    vehicle = Multirotor(copy.deepcopy(quad_params), aero=False)
    vehicle.set_frame(FrameConvention.FLU_ENU)
    controller = SE3Control(copy.deepcopy(quad_params))
    controller.set_frame(FrameConvention.FLU_ENU)
    imu = _imu()
    imu.set_frame(FrameConvention.FLU_ENU)
    estimator = WindUKF(copy.deepcopy(quad_params), dt=0.01)
    estimator.set_frame(FrameConvention.FLU_ENU)
    _run_simulate(world, vehicle, controller, imu, estimator)
    assert np.allclose(vehicle.weight, [0.0, 0.0, _MASS * _G])
    assert np.allclose(controller.thrust_axis, [0.0, 0.0, -1.0])
    state = hover_state()
    statedot = {"vdot": np.zeros(3), "wdot": np.zeros(3)}
    measured = imu.measurement(state, statedot, with_noise=False)
    assert np.allclose(measured["accel"], [0.0, 0.0, -9.81], atol=1e-6)
    _frd_hover_accel_is_level(estimator)


def test_simulate_replaces_an_earlier_frame_with_the_world_frame():
    world = World.empty(_EXTENTS, frame=FrameConvention.FLU_ENU)
    vehicle = Multirotor(copy.deepcopy(quad_params), aero=False)
    vehicle.set_frame(FrameConvention.FRD_NED)
    controller = SE3Control(copy.deepcopy(quad_params))
    controller.set_frame(FrameConvention.FRD_NED)
    imu = _imu()
    imu.set_frame(FrameConvention.FRD_NED)
    estimator = _FramelessEstimator()
    _run_simulate(world, vehicle, controller, imu, estimator)
    assert estimator.marker == "untouched"
    assert np.allclose(vehicle.weight, [0.0, 0.0, -_MASS * _G])
    assert np.allclose(controller.thrust_axis, [0.0, 0.0, 1.0])
    state = hover_state()
    statedot = {"vdot": np.zeros(3), "wdot": np.zeros(3)}
    measured = imu.measurement(state, statedot, with_noise=False)
    assert np.allclose(measured["accel"], [0.0, 0.0, 9.81], atol=1e-6)


def test_simulate_batch_mixed_frames_raise_before_stepping():
    worlds = [
        World.empty(_EXTENTS, frame=FrameConvention.FRD_NED),
        World.empty(_EXTENTS, frame=FrameConvention.FLU_ENU),
    ]
    initial_states, vehicle, controller, trajectory, wind, imu = _two_drone_hover()
    counter = _UpdateCounter(controller)
    with pytest.raises(ValueError):
        simulate_batch(
            worlds,
            initial_states,
            vehicle,
            counter,
            trajectory,
            wind,
            imu,
            np.full(2, 0.01),
            0.01,
            0.25,
            terminate=False,
        )
    assert counter.updates == 0
    assert np.allclose(
        vehicle.params.weight[0].detach().cpu().numpy(),
        [0.0, 0.0, -_MASS * _G],
    )


def test_simulate_batch_shared_frame_list_pushes_frd():
    worlds = [
        World.empty(_EXTENTS, frame=FrameConvention.FRD_NED),
        World.empty(_EXTENTS, frame=FrameConvention.FRD_NED),
    ]
    initial_states, vehicle, controller, trajectory, wind, imu = _two_drone_hover()
    vehicle.set_frame(FrameConvention.FLU_ENU)
    controller.set_frame(FrameConvention.FLU_ENU)
    imu.set_frame(FrameConvention.FLU_ENU)
    simulate_batch(
        worlds,
        initial_states,
        vehicle,
        controller,
        trajectory,
        wind,
        imu,
        np.full(2, 0.01),
        0.01,
        0.25,
        terminate=False,
    )
    assert np.allclose(
        vehicle.params.weight[0].detach().cpu().numpy(),
        [0.0, 0.0, _MASS * _G],
    )
    assert np.allclose(controller.thrust_axis.detach().cpu().numpy(), [0.0, 0.0, -1.0])
    assert np.allclose(imu.gravity_vector.detach().cpu().numpy(), [0.0, 0.0, _G])


def test_simulate_batch_single_world_pushes_frd():
    world = World.empty(_EXTENTS, frame=FrameConvention.FRD_NED)
    initial_states, vehicle, controller, trajectory, wind, imu = _two_drone_hover()
    vehicle.set_frame(FrameConvention.FLU_ENU)
    controller.set_frame(FrameConvention.FLU_ENU)
    imu.set_frame(FrameConvention.FLU_ENU)
    simulate_batch(
        world,
        initial_states,
        vehicle,
        controller,
        trajectory,
        wind,
        imu,
        np.full(2, 0.01),
        0.01,
        0.25,
        terminate=False,
    )
    assert np.allclose(
        vehicle.params.weight[0].detach().cpu().numpy(),
        [0.0, 0.0, _MASS * _G],
    )
    assert np.allclose(imu.gravity_vector.detach().cpu().numpy(), [0.0, 0.0, _G])


def test_run_sim_frd_hover():
    from examples.multimav_multiprocessing_demo import run_sim

    trajectory = HoverTraj(x0=np.zeros(3))
    _time, states, _controls, _flats = run_sim(
        trajectory,
        0.0,
        t_final=0.2,
        t_step=0.01,
        frame=FrameConvention.FRD_NED,
    )
    height = float(np.asarray(trajectory.x0)[2])
    assert states["x"][-1, 2] == pytest.approx(height, abs=1e-2)
    assert np.allclose(states["q"][-1], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-2)
