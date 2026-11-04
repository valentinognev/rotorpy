"""Offboard setpoint kinds the Gazebo homing tracker can send to PX4."""

from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Mapping, Optional, Tuple

import numpy as np

from rotorpy.controllers.px4_cascade.gains import GAINS
from rotorpy.controllers.px4_cascade.math import (
    attitude_rate_setpoint,
    saturate_thrust,
    thrust_from_acceleration,
    thrust_to_attitude,
)
from rotorpy.guidance.frames import ned_to_sim, sim_to_ned
from rotorpy.guidance.homing import track_acceleration


class CommandMode(str, Enum):
    ATTITUDE = "attitude"
    RATE = "rate"
    VELOCITY = "velocity"


def parse_command_mode(value: str) -> CommandMode:
    try:
        return CommandMode(value)
    except ValueError:
        raise ValueError(
            "command_mode must be one of: {}".format(
                ", ".join(mode.value for mode in CommandMode)
            )
        ) from None


OFFBOARD_PX4_PARAMETERS: Tuple[str, ...] = (
    "MPC_THR_HOVER",
    "MPC_THR_MIN",
    "MPC_THR_MAX",
    "MPC_THR_XY_MARG",
    "MPC_TILTMAX_AIR",
    "MC_ROLL_P",
    "MC_PITCH_P",
    "MC_YAW_P",
    "MC_YAW_WEIGHT",
    "MC_ROLLRATE_MAX",
    "MC_PITCHRATE_MAX",
    "MC_YAWRATE_MAX",
)


@dataclass(frozen=True)
class OffboardGains:
    hover_thrust: float
    thr_min: float
    thr_max: float
    thr_xy_margin: float
    tilt_max_rad: float
    roll_p: float
    pitch_p: float
    yaw_p: float
    yaw_weight: float
    rate_limit_rad: np.ndarray
    track_tau_s: float
    max_tilt_deg: float
    min_specific_up_g: float


@dataclass(frozen=True)
class AttitudeSetpoint:
    # Scalar-last quaternion with w >= 0, in the caller's NED/FRD frame.
    quat_xyzw: np.ndarray
    # Normalized collective thrust in [0, 1]; equals hover thrust at zero error.
    thrust: float
    body_rates_frd: Optional[np.ndarray] = None


def offboard_gains(
    config: Mapping, px4_params: Mapping[str, float]
) -> Tuple[OffboardGains, List[str]]:
    """Gains from live PX4 parameters; unanswered ones fall back to `GAINS`.

    Returns the gains and the names that fell back. PX4 tilt and rate limits are
    in degrees; the gains hold radians.
    """
    fallback: List[str] = []
    values = {}
    for name in OFFBOARD_PX4_PARAMETERS:
        raw = px4_params.get(name)
        if raw is None or not np.isfinite(float(raw)):
            fallback.append(name)
            values[name] = float(GAINS[name])
        else:
            values[name] = float(raw)
    gains = OffboardGains(
        hover_thrust=values["MPC_THR_HOVER"],
        thr_min=values["MPC_THR_MIN"],
        thr_max=values["MPC_THR_MAX"],
        thr_xy_margin=values["MPC_THR_XY_MARG"],
        tilt_max_rad=float(np.deg2rad(values["MPC_TILTMAX_AIR"])),
        roll_p=values["MC_ROLL_P"],
        pitch_p=values["MC_PITCH_P"],
        yaw_p=values["MC_YAW_P"],
        yaw_weight=values["MC_YAW_WEIGHT"],
        rate_limit_rad=np.deg2rad(
            [
                values["MC_ROLLRATE_MAX"],
                values["MC_PITCHRATE_MAX"],
                values["MC_YAWRATE_MAX"],
            ]
        ),
        track_tau_s=float(config["track_tau_s"]),
        max_tilt_deg=float(config["max_tilt_deg"]),
        min_specific_up_g=float(config["min_specific_up_g"]),
    )
    return gains, fallback


def _finite_vector(name: str, value) -> np.ndarray:
    vector = np.array(value, dtype=float, copy=True)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError("{} must be a finite 3-vector".format(name))
    return vector


def attitude_setpoint(
    velocity_cmd_ned,
    acceleration_cmd_ned,
    yaw_rad: float,
    measured_velocity_ned,
    gains: OffboardGains,
    enable_acceleration_feedforward: bool = False,
) -> AttitudeSetpoint:
    """Attitude and thrust that track a velocity command.

    All vectors and the yaw share one NED frame; the quaternion comes back in it.
    """
    velocity_cmd = _finite_vector("velocity_cmd_ned", velocity_cmd_ned)
    acceleration_cmd = _finite_vector("acceleration_cmd_ned", acceleration_cmd_ned)
    measured = _finite_vector("measured_velocity_ned", measured_velocity_ned)
    if not np.isfinite(yaw_rad):
        raise ValueError("yaw_rad must be finite")

    feedforward = acceleration_cmd if enable_acceleration_feedforward else np.zeros(3)
    # track_acceleration takes simulator-world vectors (north, east, up).
    acc_sim = track_acceleration(
        ned_to_sim(measured),
        ned_to_sim(velocity_cmd),
        ned_to_sim(feedforward),
        tau_s=gains.track_tau_s,
        max_tilt_deg=gains.max_tilt_deg,
        min_specific_up_g=gains.min_specific_up_g,
    )
    thrust_vector = thrust_from_acceleration(
        sim_to_ned(acc_sim), gains.hover_thrust, gains.tilt_max_rad, gains.thr_min
    )
    thrust_vector = saturate_thrust(thrust_vector, gains.thr_max, gains.thr_xy_margin)
    quaternion, thrust_body_z = thrust_to_attitude(thrust_vector, float(yaw_rad))
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return AttitudeSetpoint(
        quat_xyzw=np.array(quaternion, dtype=float),
        thrust=float(-thrust_body_z),
    )


def rate_setpoint(
    setpoint: AttitudeSetpoint, current_quat_xyzw, gains: OffboardGains
) -> AttitudeSetpoint:
    """Copy of `setpoint` with PX4 attitude-controller body rates (FRD rad/s).

    Both quaternions are scalar-last in the same NED frame.
    """
    current = np.array(current_quat_xyzw, dtype=float, copy=True)
    norm = float(np.linalg.norm(current)) if np.all(np.isfinite(current)) else float("nan")
    if current.shape != (4,) or not np.isfinite(norm) or norm == 0.0:
        raise ValueError("current_quat_xyzw must be a finite, non-zero 4-vector")
    rates = attitude_rate_setpoint(
        current / norm,
        setpoint.quat_xyzw,
        0.0,
        gains.roll_p,
        gains.pitch_p,
        gains.yaw_p,
        gains.yaw_weight,
        gains.rate_limit_rad,
    )
    return replace(setpoint, body_rates_frd=rates)
