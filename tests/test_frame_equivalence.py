"""Five-second FLU-ENU versus FRD-NED flights.

Each test fails if the world frame is ignored or the FRD record, mapped
back, misses the FLU end state by more than the global tolerances.
"""
import copy
import os

import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation

from rotorpy.controllers.quadrotor_control import BatchedSE3Control, SE3Control
from rotorpy.environments import Environment
from rotorpy.frames import (
    FrameConvention,
    aabb_enu_to_ned,
    attitude_enu_flu_to_ned_frd,
    attitude_ned_frd_to_enu_flu,
    body_to_sensor_flu_to_frd,
    enu_to_ned,
    flat_enu_to_ned,
    flu_to_frd,
    frd_to_flu,
    ned_to_enu,
    yaw_enu_to_ned,
)
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.sensors.imu import BatchedImu, Imu
from rotorpy.simulate import simulate_batch
from rotorpy.trajectories.circular_traj import ThreeDCircularTraj
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.trajectories.minsnap import BatchedMinSnap, MinSnap
from rotorpy.utils.trajgen_utils import generate_random_minsnap_traj
from rotorpy.vehicles.crazyflie_params import quad_params as crazyflie_params
from rotorpy.vehicles.hummingbird_params import quad_params as hummingbird_params
from rotorpy.vehicles.multirotor import BatchedMultirotor, BatchedMultirotorParams, Multirotor
from rotorpy.wind.default_winds import BatchedNoWind, SinusoidWind
from rotorpy.world import World

_POS_ATOL = 1e-2
_VEL_ATOL = 1e-2
_ATT_ATOL = 1e-2
_RATE_ATOL = 1e-2
_KEYPOINT_ATOL = 2.0
_DURATION = 5.0
_HOVER_ROTORS = np.array([1788.53, 1788.53, 1788.53, 1788.53])


class _FlatEnuToNed(object):
    """FRD flat outputs are the ENU flat outputs mapped by Task 1."""

    def __init__(self, trajectory):
        self._trajectory = trajectory

    def update(self, t):
        return flat_enu_to_ned(self._trajectory.update(t))


class _WindEnuToNed(object):
    """FRD wind samples are the ENU samples mapped by Task 1."""

    def __init__(self, wind):
        self._wind = wind

    def update(self, t, position):
        sample = self._wind.update(t, ned_to_enu(position))
        return enu_to_ned(sample)


def _quiet_sensor_params():
    return {
        "initial_bias": np.zeros(3),
        "noise_density": np.zeros(3),
        "random_walk": np.zeros(3),
    }


def _quiet_imu(sampling_rate):
    return Imu(
        accelerometer_params=_quiet_sensor_params(),
        gyroscope_params=_quiet_sensor_params(),
        sampling_rate=sampling_rate,
    )


def _quiet_batched_imu(num_drones, device, sampling_rate):
    accel = {
        "initial_bias": torch.zeros(3, dtype=torch.double, device=device),
        "noise_density": torch.zeros(3, dtype=torch.double, device=device),
        "random_walk": torch.zeros(3, dtype=torch.double, device=device),
    }
    gyro = {
        "initial_bias": torch.zeros(3, dtype=torch.double, device=device),
        "noise_density": torch.zeros(3, dtype=torch.double, device=device),
        "random_walk": torch.zeros(3, dtype=torch.double, device=device),
    }
    return BatchedImu(
        num_drones,
        accelerometer_params=accel,
        gyroscope_params=gyro,
        sampling_rate=sampling_rate,
        device=device,
    )


def _basic_initial_state():
    return {
        "x": np.array([0.0, 0.0, 0.0]),
        "v": np.zeros(3),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.zeros(3),
        "wind": np.array([0.0, 0.0, 0.0]),
        "rotor_speeds": np.array(_HOVER_ROTORS, copy=True),
    }


def _state_to_frd(state):
    return {
        "x": enu_to_ned(state["x"]),
        "v": enu_to_ned(state["v"]),
        "q": attitude_enu_flu_to_ned_frd(state["q"]),
        "w": flu_to_frd(state["w"]),
        "wind": enu_to_ned(state["wind"]),
        "rotor_speeds": np.array(state["rotor_speeds"], dtype=float, copy=True),
    }


def _frd_world(world):
    """Copy extents and blocks through aabb_enu_to_ned. The file is not rewritten."""
    data = copy.deepcopy(world.world)
    data["bounds"] = dict(data["bounds"])
    data["bounds"]["extents"] = aabb_enu_to_ned(data["bounds"]["extents"])
    blocks = []
    for block in data.get("blocks", []):
        copied = dict(block)
        copied["extents"] = aabb_enu_to_ned(block["extents"])
        blocks.append(copied)
    data["blocks"] = blocks
    return World(data, frame=FrameConvention.FRD_NED)


def _double_pillar_world():
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "rotorpy", "worlds", "double_pillar.json")
    )
    return World.from_file(path)


def _attitude_error(q_got, q_want):
    got = Rotation.from_quat(np.asarray(q_got, dtype=float).reshape(4))
    want = Rotation.from_quat(np.asarray(q_want, dtype=float).reshape(4))
    return float((want.inv() * got).magnitude())


def _assert_mapped_end(flu_state, frd_state):
    position = ned_to_enu(frd_state["x"])
    velocity = ned_to_enu(frd_state["v"])
    attitude = attitude_ned_frd_to_enu_flu(frd_state["q"])
    rate = frd_to_flu(frd_state["w"])
    np.testing.assert_allclose(position, flu_state["x"], rtol=0.0, atol=_POS_ATOL)
    np.testing.assert_allclose(velocity, flu_state["v"], rtol=0.0, atol=_VEL_ATOL)
    attitude_error = _attitude_error(attitude, flu_state["q"])
    assert attitude_error <= _ATT_ATOL, "attitude error %g rad exceeds %g" % (
        attitude_error,
        _ATT_ATOL,
    )
    np.testing.assert_allclose(rate, flu_state["w"], rtol=0.0, atol=_RATE_ATOL)


def _assert_full_duration(timestamp):
    """t_final is 5 s. Binary steps of 0.01 s land on the first sample at or just after 5."""
    assert timestamp + 1e-9 >= _DURATION
    assert timestamp < _DURATION + 0.02


def _last_single(result):
    state = result["state"]
    index = -1
    return {
        "x": state["x"][index],
        "v": state["v"][index],
        "q": state["q"][index],
        "w": state["w"][index],
        "time": result["time"][index],
    }


@pytest.mark.frame_equivalence
def test_basic_usage_matches():
    """Crazyflie circle in the double-pillar world, FLU-ENU against FRD-NED."""
    trajectory = ThreeDCircularTraj(radius=np.array([3.0, 3.0, 0.0]), freq=np.array([0.15, 0.15, 0.0]))
    flu_world = _double_pillar_world()
    frd_world = _frd_world(flu_world)
    flu_state0 = _basic_initial_state()
    flu = _run_environment(flu_world, trajectory, SinusoidWind(), flu_state0, crazyflie_params)
    frd = _run_environment(
        frd_world,
        _FlatEnuToNed(trajectory),
        _WindEnuToNed(SinusoidWind()),
        _state_to_frd(flu_state0),
        crazyflie_params,
    )
    flu_end = _last_single(flu)
    frd_end = _last_single(frd)
    _assert_full_duration(flu_end["time"])
    _assert_full_duration(frd_end["time"])
    assert flu_end["time"] == pytest.approx(frd_end["time"], abs=1e-12)
    _assert_mapped_end(flu_end, frd_end)


def _run_environment(world, trajectory, wind, initial_state, params):
    vehicle = Multirotor(params)
    controller = SE3Control(params)
    environment = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=trajectory,
        wind_profile=wind,
        sim_rate=100,
        imu=_quiet_imu(100),
        world=world,
    )
    vehicle.initial_state = initial_state
    return environment.run(
        t_final=_DURATION,
        use_mocap=False,
        terminate=False,
        plot=False,
        plot_mocap=False,
        plot_estimator=False,
        plot_imu=False,
        animate_bool=False,
        animate_wind=False,
        verbose=False,
    )


@pytest.mark.frame_equivalence
def test_run_sim_matches():
    """One Lissajous through run_sim, FLU frame against a wrapped FRD frame."""
    from examples.multimav_multiprocessing_demo import run_sim

    trajectory = TwoDLissajous(A=1, B=1, a=2, b=1, x_offset=-0.5, y_offset=0, height=2.0)
    _time_flu, states_flu, _controls_flu, _flats_flu = run_sim(
        trajectory,
        0.0,
        t_final=_DURATION,
        t_step=0.01,
        frame=FrameConvention.FLU_ENU,
    )
    _time_frd, states_frd, _controls_frd, _flats_frd = run_sim(
        _FlatEnuToNed(trajectory),
        0.0,
        t_final=_DURATION,
        t_step=0.01,
        frame=FrameConvention.FRD_NED,
    )
    _assert_full_duration(_time_flu[-1])
    _assert_full_duration(_time_frd[-1])
    assert _time_flu[-1] == pytest.approx(_time_frd[-1], abs=1e-12)
    flu_end = {key: states_flu[key][-1] for key in ("x", "v", "q", "w")}
    frd_end = {key: states_frd[key][-1] for key in ("x", "v", "q", "w")}
    _assert_mapped_end(flu_end, frd_end)


def _minsnap_in_ned(trajectory):
    points = np.vstack([enu_to_ned(point) for point in trajectory.points])
    yaw = np.asarray(trajectory.yaw, dtype=float).reshape(-1)
    assert yaw.shape[0] == points.shape[0]
    ned_yaw = np.array([yaw_enu_to_ned(float(angle)) for angle in yaw])
    return MinSnap(points, yaw_angles=ned_yaw, v_avg=trajectory.v_avg, verbose=False)


def _batch_initial_states(num_drones, quaternion, device):
    rotors = torch.tensor(_HOVER_ROTORS, dtype=torch.double, device=device).repeat(num_drones, 1)
    return {
        "x": torch.zeros(num_drones, 3, dtype=torch.double, device=device),
        "v": torch.zeros(num_drones, 3, dtype=torch.double, device=device),
        "q": torch.tensor(quaternion, dtype=torch.double, device=device).repeat(num_drones, 1),
        "w": torch.zeros(num_drones, 3, dtype=torch.double, device=device),
        "wind": torch.zeros(num_drones, 3, dtype=torch.double, device=device),
        "rotor_speeds": rotors,
    }


def _run_batch(world, trajectories, initial_states, params_list, device):
    num_drones = len(params_list)
    batch_params = BatchedMultirotorParams(params_list, num_drones, device)
    vehicle = BatchedMultirotor(
        batch_params,
        num_drones,
        initial_states,
        device=device,
        control_abstraction="cmd_ctatt",
        integrator="dopri5",
    )
    controller = BatchedSE3Control(batch_params, num_drones, device=device)
    batched = BatchedMinSnap(trajectories, device=device)
    wind = BatchedNoWind(num_drones)
    imu = _quiet_batched_imu(num_drones, device, sampling_rate=100)
    t_final = np.full(num_drones, _DURATION)
    return simulate_batch(
        world,
        initial_states,
        vehicle,
        controller,
        batched,
        wind,
        imu,
        t_final,
        0.01,
        0.25,
        terminate=False,
        print_fps=False,
    )


@pytest.mark.frame_equivalence
def test_batched_matches():
    """Two seeded MinSnap drones, Crazyflie then Hummingbird, for 5 s."""
    device = torch.device("cpu")
    world = World({"bounds": {"extents": [-10, 10, -10, 10, -10, 10]}, "blocks": []})
    np.random.seed(10)
    trajectories = []
    num_done = 0
    while num_done < 2:
        traj = generate_random_minsnap_traj(
            world,
            4,
            2,
            min_distance=1,
            max_distance=2,
            start_position=np.zeros(3),
        )
        if traj is not None:
            trajectories.append(traj)
            num_done += 1
    frd_trajectories = [_minsnap_in_ned(traj) for traj in trajectories]
    cf_params = copy.deepcopy(crazyflie_params)
    hb_params = copy.deepcopy(hummingbird_params)
    cf_params["motor_noise_std"] = 0
    hb_params["motor_noise_std"] = 0
    params_list = [cf_params, hb_params]
    flu_initial = _batch_initial_states(2, np.array([0.0, 0.0, 0.0, 1.0]), device)
    frd_quaternion = attitude_enu_flu_to_ned_frd(np.array([0.0, 0.0, 0.0, 1.0]))
    frd_initial = _batch_initial_states(2, frd_quaternion, device)

    flu_result = _run_batch(world, trajectories, flu_initial, params_list, device)
    frd_result = _run_batch(
        _frd_world(world),
        frd_trajectories,
        frd_initial,
        copy.deepcopy(params_list),
        device,
    )
    flu_time, flu_state = flu_result[0], flu_result[1]
    frd_time, frd_state = frd_result[0], frd_result[1]
    flu_exit = flu_result[7]
    frd_exit = frd_result[7]
    for drone in range(2):
        assert int(flu_exit[drone]) == int(frd_exit[drone])
        index = int(flu_exit[drone]) - 1
        _assert_full_duration(flu_time[index, drone])
        _assert_full_duration(frd_time[index, drone])
        assert flu_time[index, drone] == pytest.approx(frd_time[index, drone], abs=1e-12)
        flu_end = {key: flu_state[key][index, drone] for key in ("x", "v", "q", "w")}
        frd_end = {key: frd_state[key][index, drone] for key in ("x", "v", "q", "w")}
        _assert_mapped_end(flu_end, frd_end)


class _InwardYaw(object):
    """Time-based inward yaw from examples/camera_demo.py."""

    def __init__(self, freq):
        self.omega = 2.0 * np.pi * np.array(freq, dtype=float)
        self._prev_t = None
        self._prev_yaw_raw = 0.0

    def update(self, t, x_dot=None):
        yaw = np.arctan2(-np.sin(self.omega[1] * t), -np.cos(self.omega[0] * t))
        if self._prev_t is not None and t > self._prev_t:
            dt = t - self._prev_t
            dyaw = yaw - self._prev_yaw_raw
            dyaw = (dyaw + np.pi) % (2.0 * np.pi) - np.pi
            yaw_dot = dyaw / dt
        else:
            yaw_dot = 0.0
        self._prev_t = t
        self._prev_yaw_raw = yaw
        return {"yaw": yaw, "yaw_dot": yaw_dot, "yaw_ddot": 0.0}


def _demo_circle():
    return (
        np.array([0.0, 0.0, 0.0]),
        np.array([3.0, 3.0, 0.0]),
        np.array([0.2, 0.2, 0.0]),
    )


def _demo_trajectory():
    center, radius, freq = _demo_circle()
    return ThreeDCircularTraj(center=center, radius=radius, freq=freq, yaw_traj=_InwardYaw(freq))


def _demo_extrinsics():
    base_orientation = (
        Rotation.from_euler("x", -45, degrees=True)
        * Rotation.from_euler("y", -90, degrees=True)
        * Rotation.from_euler("x", 90, degrees=True)
    )
    return {
        "position": np.array([0.0, 0.0, 0.0]),
        "orientation": base_orientation.as_quat(),
    }


def _demo_intrinsics():
    return {
        "fx": 400.0,
        "fy": 400.0,
        "width": 640,
        "height": 480,
        "cx": 320.0,
        "cy": 240.0,
        "dist_coeffs": np.array([-0.3, 0.1, 0.0, 0.0, 0.0]),
    }


def _frd_extrinsics(extrinsics):
    return {
        "position": flu_to_frd(extrinsics["position"]),
        "orientation": body_to_sensor_flu_to_frd(extrinsics["orientation"]),
    }


def _camera_frd_world(flu_world):
    """Convert extents, blocks, and feature points. The generator does not run again."""
    data = copy.deepcopy(flu_world.world)
    data["bounds"] = dict(data["bounds"])
    data["bounds"]["extents"] = aabb_enu_to_ned(data["bounds"]["extents"])
    blocks = []
    for block in data.get("blocks", []):
        copied = dict(block)
        copied["extents"] = aabb_enu_to_ned(block["extents"])
        blocks.append(copied)
    data["blocks"] = blocks
    points = np.asarray(flu_world.get_surface_features(), dtype=float)
    data["features"] = {"points": np.vstack([enu_to_ned(point) for point in points])}
    return World(data, frame=FrameConvention.FRD_NED)


def _assert_keypoints_within_two_pixels(flu_keypoints, frd_keypoints):
    flu = np.asarray(flu_keypoints, dtype=float)
    frd = np.asarray(frd_keypoints, dtype=float)
    if flu.shape != frd.shape or flu.ndim != 2 or flu.shape[1] != 2 or flu.shape[0] == 0:
        raise AssertionError(
            "FRD keypoints differ by more than 2 pixels (shapes %s vs %s)"
            % (getattr(frd, "shape", None), getattr(flu, "shape", None))
        )
    delta = float(np.max(np.abs(flu - frd)))
    assert delta <= _KEYPOINT_ATOL, "FRD keypoints differ by %g pixels, which is more than 2" % delta


def _demo_initial_state():
    _center, _radius, freq = _demo_circle()
    yaw = _InwardYaw(freq).update(0.0)["yaw"]
    return {
        "x": np.array([0.0, 0.0, 0.0]),
        "v": np.zeros(3),
        "q": Rotation.from_euler("z", yaw).as_quat(),
        "w": np.zeros(3),
        "wind": np.array([0.0, 0.0, 0.0]),
        "rotor_speeds": np.array(_HOVER_ROTORS, copy=True),
    }


def _run_camera_demo(world, trajectory, initial_state, extrinsics):
    camera = PinholeCamera(
        intrinsics=_demo_intrinsics(),
        extrinsics=extrinsics,
        frame_rate=25,
        splat_radius=3,
        feature_output="rgb",
    )
    vehicle = Multirotor(crazyflie_params)
    controller = SE3Control(crazyflie_params)
    environment = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=trajectory,
        wind_profile=None,
        sim_rate=100,
        imu=_quiet_imu(100),
        world=world,
        camera=camera,
    )
    vehicle.initial_state = initial_state
    return environment.run(
        t_final=_DURATION,
        use_mocap=False,
        terminate=False,
        plot=False,
        plot_mocap=False,
        plot_estimator=False,
        plot_imu=False,
        plot_camera=False,
        animate_bool=False,
        animate_wind=False,
        verbose=False,
    )


@pytest.mark.frame_equivalence
def test_camera_demo_keypoints_match():
    """Last camera frame of the double-pillar circle, FLU pixels against FRD pixels."""
    flu_world = _double_pillar_world()
    flu_world.generate_surface_features(mode="regular", spacing=0.08)
    frd_world = _camera_frd_world(flu_world)
    flu_state0 = _demo_initial_state()
    flu = _run_camera_demo(flu_world, _demo_trajectory(), flu_state0, _demo_extrinsics())
    frd = _run_camera_demo(
        frd_world,
        _FlatEnuToNed(_demo_trajectory()),
        _state_to_frd(flu_state0),
        _frd_extrinsics(_demo_extrinsics()),
    )
    _assert_full_duration(flu["time"][-1])
    _assert_full_duration(frd["time"][-1])
    flu_keypoints = flu["camera_measurements"]["keypoints"][-1]
    frd_keypoints = frd["camera_measurements"]["keypoints"][-1]
    _assert_keypoints_within_two_pixels(flu_keypoints, frd_keypoints)


@pytest.mark.frame_equivalence
def test_camera_visualization_keypoints_match():
    """Parked regular-feature render. The lens-noise grid is not rendered."""
    from examples.camera_visualization import BASE_INTRINSICS, EXTRINSICS, STATE, WORLD_DATA

    flu_world = World(
        copy.deepcopy(WORLD_DATA),
        add_features=True,
        feature_mode="regular",
        feature_spacing=0.25,
    )
    frd_world = _camera_frd_world(flu_world)
    intrinsics = dict(BASE_INTRINSICS, fx=480.0, fy=480.0, dist_coeffs=np.array([-0.20, 0.05, 0.0, 0.0, 0.0]))
    flu_camera = PinholeCamera(intrinsics=intrinsics, extrinsics=EXTRINSICS, feature_output="rgb")
    frd_camera = PinholeCamera(
        intrinsics=intrinsics,
        extrinsics=_frd_extrinsics(EXTRINSICS),
        feature_output="rgb",
    )
    frd_state = {
        "x": enu_to_ned(STATE["x"]),
        "q": attitude_enu_flu_to_ned_frd(STATE["q"]),
    }
    flu_keypoints = flu_camera.measurement(STATE, flu_world)["keypoints"]
    frd_keypoints = frd_camera.measurement(frd_state, frd_world)["keypoints"]
    _assert_keypoints_within_two_pixels(flu_keypoints, frd_keypoints)
