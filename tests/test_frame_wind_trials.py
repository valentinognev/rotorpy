"""Paired FLU-ENU and FRD-NED wind-filter trials.

``test_thrust_removal_follows_axis`` fails if commanded thrust is removed on
body +z regardless of ``thrust_axis``. ``test_fifty_trials_match`` fails if a
FRD-NED trial, mapped back, misses the FLU-ENU trial.
"""
import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from examples.wind_filter_example import (
    P0,
    Q,
    auto_system_identification,
    calibration_traj,
    points,
    remove_commanded_thrust,
)
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.environments import Environment
from rotorpy.estimators.wind_ukf import WindUKF
from rotorpy.frames import (
    FrameConvention,
    attitude_enu_flu_to_ned_frd,
    attitude_ned_frd_to_enu_flu,
    enu_to_ned,
    flat_enu_to_ned,
    flu_to_frd,
    frd_to_flu,
    ned_to_enu,
    yaw_enu_to_ned,
)
from rotorpy.trajectories.hover_traj import HoverTraj
from rotorpy.trajectories.minsnap import MinSnap
from rotorpy.utils.postprocessing import unpack_sim_data
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.wind.dryden_winds import DrydenGust
from rotorpy.world import World

_NUM_TRIALS = 50
_G = 9.81
_DRAG_ATOL = 1e-3
_WIND_ATOL = 1e-2
_POS_ATOL = 1e-2
_VEL_ATOL = 1e-2
_ATT_ATOL = 1e-2
_RATE_ATOL = 1e-2
_IDENTITY_Q = np.array([0.0, 0.0, 0.0, 1.0])
_FRD_Q = attitude_enu_flu_to_ned_frd(_IDENTITY_Q)


class _FlatEnuToNed(object):
    """FRD flat outputs are the ENU flat outputs mapped by Task 1."""

    def __init__(self, trajectory):
        self._trajectory = trajectory

    def update(self, t):
        return flat_enu_to_ned(self._trajectory.update(t))


class _ReplayWind(object):
    """Replay a world-frame wind series.

    ``DrydenGust.update`` draws three ``uniform(-1, 1)`` samples per step.
    Those draws are burned here so the IMU and mocap streams stay aligned
    with the FLU-ENU evaluation.
    """

    def __init__(self, samples):
        self._samples = np.array(samples, dtype=float, copy=True)
        self._index = 0

    def update(self, t, position):
        np.random.uniform(-1.0, 1.0)
        np.random.uniform(-1.0, 1.0)
        np.random.uniform(-1.0, 1.0)
        sample = self._samples[self._index]
        self._index += 1
        return np.array(sample, dtype=float, copy=True)


def _frd_world():
    """Same empty box the environment builds when world is left unset."""
    bound = 3
    return World.empty(
        (-bound, bound, -bound, bound, -bound, bound),
        frame=FrameConvention.FRD_NED,
    )


def _frd_calibration():
    """Waypoints are ``enu_to_ned`` of the script's points. Yaw follows."""
    ned_points = np.vstack([enu_to_ned(point) for point in points])
    yaw = np.full(ned_points.shape[0], yaw_enu_to_ned(0.0))
    return MinSnap(ned_points, yaw_angles=yaw, v_avg=1, verbose=False)


def _draw_trials():
    """Seed 0, then the script's 50 scalar and wind draws, before any flight."""
    np.random.seed(0)
    draws = []
    for _ in range(_NUM_TRIALS):
        params = copy.deepcopy(quad_params)
        params["mass"] *= np.random.uniform(low=0.75, high=1.25)
        params["c_Dx"] *= np.random.uniform(low=0, high=2)
        params["c_Dy"] *= np.random.uniform(low=0, high=2)
        params["c_Dz"] *= np.random.uniform(low=0, high=2)
        params["k_d"] *= np.random.uniform(low=0, high=10)
        params["k_z"] *= np.random.uniform(low=0, high=10)
        mean_flu = np.array(
            [
                np.random.uniform(low=-3, high=3),
                np.random.uniform(low=-3, high=3),
                np.random.uniform(low=-3, high=3),
            ],
            dtype=float,
        )
        sig_flu = np.array(
            [
                np.random.uniform(low=30, high=60),
                np.random.uniform(low=30, high=60),
                np.random.uniform(low=30, high=60),
            ],
            dtype=float,
        )
        draws.append(
            {
                "params": params,
                "mean_flu": mean_flu,
                "mean_frd": enu_to_ned(mean_flu),
                "sig_flu": sig_flu,
                "sig_frd": np.array([sig_flu[1], sig_flu[0], sig_flu[2]], dtype=float),
            }
        )
    return draws


def _initial_state(trajectory, params, quaternion):
    hover_speed = np.sqrt(params["mass"] * _G / (4.0 * params["k_eta"]))
    return {
        "x": np.array(trajectory.update(0)["x"], dtype=float, copy=True),
        "v": np.zeros(3),
        "q": np.array(quaternion, dtype=float, copy=True),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.ones(4) * hover_speed,
    }


def _run(params, trajectory, world, estimator, wind, quaternion, t_final):
    vehicle = Multirotor(params)
    controller = SE3Control(params)
    environment = Environment(
        vehicle=vehicle,
        controller=controller,
        trajectory=trajectory,
        wind_profile=wind,
        estimator=estimator,
        world=world,
    )
    environment.vehicle.initial_state = _initial_state(trajectory, params, quaternion)
    result = environment.run(t_final=t_final, plot=False, verbose=False)
    return environment.vehicle, result


def _fitted_drag(result, mass, thrust_axis):
    data = unpack_sim_data(result)
    orientation = data[["qx", "qy", "qz", "qw"]].to_numpy()
    velocity = (data[["xdot", "ydot", "zdot"]].to_numpy())[..., np.newaxis]
    rotation = Rotation.from_quat(orientation).as_matrix()
    body_velocity = (np.transpose(rotation, axes=[0, 2, 1]) @ velocity)[:, :, 0]
    accel = data[["ax", "ay", "az"]].to_numpy(copy=True)
    accel = remove_commanded_thrust(
        accel, data["thrustdes"].to_numpy(), mass, thrust_axis
    )
    return np.array(
        auto_system_identification(mass, body_velocity, np.abs(accel), plot=False),
        dtype=float,
    )


def _wind_rmse(result):
    data = unpack_sim_data(result)
    orientation = data[["qx", "qy", "qz", "qw"]].to_numpy()
    wind = (data[["windx", "windy", "windz"]].to_numpy())[..., np.newaxis]
    rotation = Rotation.from_quat(orientation).as_matrix()
    body_wind = (np.transpose(rotation, axes=[0, 2, 1]) @ wind)[:, :, 0]
    estimate = data[["xhat_6", "xhat_7", "xhat_8"]].to_numpy()
    return np.sqrt(((estimate - body_wind) ** 2).mean(axis=0))


def _ukf(params, drag, xhat0):
    estimator_params = copy.deepcopy(params)
    estimator_params["c_Dx"] = drag[0]
    estimator_params["c_Dy"] = drag[1]
    estimator_params["c_Dz"] = drag[2]
    return WindUKF(
        estimator_params,
        Q=np.array(Q, copy=True),
        xhat0=np.array(xhat0, dtype=float, copy=True),
        P0=np.array(P0, copy=True),
        dt=1.0 / 100.0,
        alpha=1e-3,
        beta=2,
        kappa=-1,
    )


def _xhat0_flu(mean_flu):
    return np.array(
        [0.0, 0.0, 0.0, 0.01, 0.01, 0.01, mean_flu[0], mean_flu[1], mean_flu[2]],
        dtype=float,
    )


def _xhat0_frd(mean_flu):
    """Level FRD hover is the trim, so the small-angle Euler state starts at 0."""
    body_velocity = flu_to_frd(np.array([0.01, 0.01, 0.01]))
    body_wind = flu_to_frd(mean_flu)
    return np.concatenate([np.zeros(3), body_velocity, body_wind])


class _RecordingWind(object):
    """Save every wind-profile update, including the t=0 sample.

    ``simulate`` overwrites ``state[-1]["wind"]`` with the next sample before
    integrating, so the logged series drops the first gust and repeats the
    last one. The vehicle and the t=0 IMU see the raw update sequence.
    """

    def __init__(self, wind):
        self._wind = wind
        self.samples = []

    def update(self, t, position):
        sample = np.array(self._wind.update(t, position), dtype=float, copy=True)
        self.samples.append(sample)
        return sample


def _replay_of(samples):
    """NED replay of a FLU wind-update series. Burns Dryden's three uniforms."""
    converted = np.vstack([enu_to_ned(sample) for sample in samples])
    return _ReplayWind(converted)


def _last_state(result):
    state = result["state"]
    return {
        "x": state["x"][-1],
        "v": state["v"][-1],
        "q": state["q"][-1],
        "w": state["w"][-1],
    }


def _attitude_error(q_got, q_want):
    got = Rotation.from_quat(np.asarray(q_got, dtype=float).reshape(4))
    want = Rotation.from_quat(np.asarray(q_want, dtype=float).reshape(4))
    return float((want.inv() * got).magnitude())


def _compare_trial(index, draw, flu_trajectory, frd_trajectory, frd_world):
    params = draw["params"]
    flu_params = copy.deepcopy(params)
    frd_params = copy.deepcopy(params)
    rng_state = np.random.get_state()

    _flu_cal_vehicle, flu_cal = _run(
        flu_params,
        flu_trajectory,
        None,
        WindUKF(flu_params, P0=np.eye(9)),
        None,
        _IDENTITY_Q,
        30,
    )
    flu_drag = _fitted_drag(flu_cal, flu_params["mass"], _flu_cal_vehicle.thrust_axis)
    flu_recorder = _RecordingWind(
        DrydenGust(dt=1.0 / 100.0, avg_wind=draw["mean_flu"], sig_wind=draw["sig_flu"])
    )
    flu_eval_vehicle, flu_eval = _run(
        flu_params,
        HoverTraj(),
        None,
        _ukf(flu_params, flu_drag, _xhat0_flu(draw["mean_flu"])),
        flu_recorder,
        _IDENTITY_Q,
        10,
    )
    flu_rmse = _wind_rmse(flu_eval)

    np.random.set_state(rng_state)
    frd_cal_vehicle, frd_cal = _run(
        frd_params,
        frd_trajectory,
        frd_world,
        WindUKF(frd_params, P0=np.eye(9)),
        None,
        _FRD_Q,
        30,
    )
    assert np.allclose(frd_cal_vehicle.thrust_axis, [0.0, 0.0, -1.0])
    assert np.allclose(flu_eval_vehicle.thrust_axis, [0.0, 0.0, 1.0])
    frd_drag = _fitted_drag(frd_cal, frd_params["mass"], frd_cal_vehicle.thrust_axis)
    _frd_eval_vehicle, frd_eval = _run(
        frd_params,
        _FlatEnuToNed(HoverTraj()),
        frd_world,
        _ukf(frd_params, frd_drag, _xhat0_frd(draw["mean_flu"])),
        _replay_of(flu_recorder.samples),
        _FRD_Q,
        10,
    )
    frd_rmse = _wind_rmse(frd_eval)

    np.testing.assert_allclose(
        frd_drag,
        flu_drag,
        rtol=0.0,
        atol=_DRAG_ATOL,
        err_msg="trial %d fitted drag" % index,
    )
    np.testing.assert_allclose(
        frd_rmse,
        flu_rmse,
        rtol=0.0,
        atol=_WIND_ATOL,
        err_msg="trial %d wind RMSE" % index,
    )
    flu_hat = flu_eval["state_estimate"]["filter_state"][-1, 6:9]
    frd_hat = frd_to_flu(frd_eval["state_estimate"]["filter_state"][-1, 6:9])
    np.testing.assert_allclose(
        frd_hat,
        flu_hat,
        rtol=0.0,
        atol=_WIND_ATOL,
        err_msg="trial %d final body wind" % index,
    )
    flu_end = _last_state(flu_eval)
    frd_end = _last_state(frd_eval)
    np.testing.assert_allclose(
        ned_to_enu(frd_end["x"]),
        flu_end["x"],
        rtol=0.0,
        atol=_POS_ATOL,
        err_msg="trial %d position" % index,
    )
    np.testing.assert_allclose(
        ned_to_enu(frd_end["v"]),
        flu_end["v"],
        rtol=0.0,
        atol=_VEL_ATOL,
        err_msg="trial %d velocity" % index,
    )
    attitude_error = _attitude_error(
        attitude_ned_frd_to_enu_flu(frd_end["q"]), flu_end["q"]
    )
    assert attitude_error <= _ATT_ATOL, "trial %d attitude error %g rad" % (
        index,
        attitude_error,
    )
    np.testing.assert_allclose(
        frd_to_flu(frd_end["w"]),
        flu_end["w"],
        rtol=0.0,
        atol=_RATE_ATOL,
        err_msg="trial %d body rate" % index,
    )


def test_thrust_removal_follows_axis():
    """Specific force [0, 0, 9.81], thrust 0.03 * 9.81, mass 0.03.

    Axis [0, 0, 1] yields z = 0. Axis [0, 0, -1] yields z = 19.62.
    """
    specific_force = np.array([0.0, 0.0, 9.81])
    thrust = 0.03 * 9.81
    mass = 0.03
    flu = remove_commanded_thrust(specific_force, thrust, mass, np.array([0.0, 0.0, 1.0]))
    frd = remove_commanded_thrust(specific_force, thrust, mass, np.array([0.0, 0.0, -1.0]))
    assert flu[2] == pytest.approx(0.0)
    assert frd[2] == pytest.approx(19.62)


@pytest.mark.frame_wind
def test_fifty_trials_match():
    """Fifty paired 30 s calibrations and 10 s evaluations."""
    draws = _draw_trials()
    assert len(draws) == _NUM_TRIALS
    frd_trajectory = _frd_calibration()
    frd_world = _frd_world()
    failures = []
    for index, draw in enumerate(draws):
        print("Trial %d/%d" % (index + 1, _NUM_TRIALS), flush=True)
        try:
            _compare_trial(index, draw, calibration_traj, frd_trajectory, frd_world)
        except AssertionError as exc:
            message = str(exc).strip().splitlines()
            summary = message[0]
            for line in message:
                if line.startswith("trial "):
                    summary = line
                    break
            failures.append(summary)
            print("FAIL", summary, flush=True)
    assert not failures, "\n".join(failures)
