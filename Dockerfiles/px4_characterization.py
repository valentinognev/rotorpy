"""PX4 parameter-dump and command-response characterization (plan Phase 7).

Offline-environment ruling: the PX4 SITL/Gazebo stack cannot be inspected or flown
here, so this module is a **documented, importable script** -- `dump_px4_parameters`
and `run_velocity_step_characterization` are not executed in this repository, and no
measured limit anywhere in `homing_config.py` comes from running them. What *is*
tested here, on synthetic data, is the analysis itself
(`analyze_step_response`/`analyze_velocity_tracking`): the arithmetic that would turn
a recorded command-response log into rise time/overshoot/settling-time/steady-state-
error numbers, so that a future live-flight run only has to supply the log, not
re-derive the analysis.

Usage once a live PX4/Gazebo session is available (not here):

    from pymavlink import mavutil
    from Dockerfiles.px4_characterization import (
        dump_px4_parameters, run_velocity_step_characterization, analyze_step_response,
    )

    connection = mavutil.mavlink_connection("udp:127.0.0.1:14540")
    params = dump_px4_parameters(connection)  # MPC_XY_VEL_P_ACC, MPC_XY_VEL_MAX, ...
    log = run_velocity_step_characterization(connection, send_velocity_setpoint, step_m_s=5.0)
    metrics = analyze_step_response(log.times_s, log.commanded_m_s, log.measured_m_s)
    # metrics.rise_time_s / overshoot_frac / settling_time_s / steady_state_error_m_s
    # feed a measured `homing_config.PROPOSED_VEHICLE_MAX_*` revision, replacing the
    # placeholder with a cited, reproducible number.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Sequence

import numpy as np

# Proposed, not verified: how long to wait for one PX4 parameter before giving up.
PROPOSED_PARAM_TIMEOUT_S = 2.0
# Proposed, not verified: settling band, as a fraction of the commanded step size,
# used by both `analyze_step_response` and `run_velocity_step_characterization`.
PROPOSED_SETTLING_BAND_FRAC = 0.05
# Parameters relevant to the Phase 7 vehicle envelope (speed/accel/tilt/yaw-rate/
# vertical-rate); not an exhaustive PX4 parameter list.
RELEVANT_PX4_PARAMETERS = (
    "MPC_XY_VEL_MAX",
    "MPC_Z_VEL_MAX_UP",
    "MPC_Z_VEL_MAX_DN",
    "MPC_XY_VEL_P_ACC",
    "MPC_ACC_HOR_MAX",
    "MPC_ACC_HOR",
    "MPC_JERK_MAX",
    "MPC_TILTMAX_AIR",
    "MC_YAWRATE_MAX",
    "MPC_THR_HOVER",
)


def dump_px4_parameters(
    connection: Any,
    names: Sequence[str] = RELEVANT_PX4_PARAMETERS,
    timeout_s: float = PROPOSED_PARAM_TIMEOUT_S,
) -> Dict[str, float]:
    """Request and collect `names` from a live PX4 mavlink `connection`.

    Documented, not run here: `connection` is a live `pymavlink.mavutil.mavlink_connection`
    (or anything exposing `.mav.param_request_read_send`/`.recv_match`, the same
    minimal surface `Dockerfiles/camera_homing_tracker.py` already depends on).
    Returns only the parameters that actually answered within `timeout_s` each; a
    caller that needs every name present must check the returned dict's keys.
    """
    import time

    values: Dict[str, float] = {}
    for name in names:
        connection.mav.param_request_read_send(
            connection.target_system, connection.target_component, name.encode("utf-8"), -1
        )
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            message = connection.recv_match(type="PARAM_VALUE", blocking=False)
            if message is not None and message.param_id.strip("\x00") == name:
                values[name] = float(message.param_value)
                break
    return values


@dataclass
class VelocityStepLog:
    """One recorded command-response trial: parallel arrays, seconds/(m/s)."""

    times_s: np.ndarray
    commanded_m_s: np.ndarray
    measured_m_s: np.ndarray


def run_velocity_step_characterization(
    connection: Any,
    send_velocity_setpoint: Callable[[float], None],
    step_m_s: float,
    duration_s: float = 5.0,
    sample_hz: float = 20.0,
) -> VelocityStepLog:
    """Command one velocity step and record PX4's own reported velocity against it.

    Documented, not run here (needs a live PX4/Gazebo link): `send_velocity_setpoint`
    is the caller's own mavlink send (for example `CameraHomingTracker._send`-style);
    this function only sequences it and reads back `LOCAL_POSITION_NED.v*` at
    `sample_hz` for `duration_s`. Analyze the result with `analyze_step_response`.
    """
    import time

    times = []
    commanded = []
    measured = []
    start = time.monotonic()
    period = 1.0 / float(sample_hz)
    next_sample = start
    while time.monotonic() - start < duration_s:
        send_velocity_setpoint(step_m_s)
        message = connection.recv_match(type="LOCAL_POSITION_NED", blocking=True, timeout=period)
        now = time.monotonic() - start
        if message is not None:
            times.append(now)
            commanded.append(step_m_s)
            measured.append(float(message.vx))
        next_sample += period
        sleep_s = next_sample - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)
    return VelocityStepLog(np.array(times), np.array(commanded), np.array(measured))


@dataclass
class StepResponseMetrics:
    """Classic step-response figures of merit, all in the response's own units."""

    rise_time_s: Optional[float]
    overshoot_frac: float
    settling_time_s: Optional[float]
    steady_state_error: float


def analyze_step_response(
    times_s: Sequence[float],
    commanded: Sequence[float],
    measured: Sequence[float],
    settling_band_frac: float = PROPOSED_SETTLING_BAND_FRAC,
) -> StepResponseMetrics:
    """Rise time (10%-90%), overshoot, settling time (first time inside and remaining
    inside `settling_band_frac` of the final commanded value), and steady-state error.

    Pure arithmetic on a recorded (or synthetic) trial: no mavlink/PX4/Gazebo
    dependency, so this is exactly what a future live-flight log plugs into. Assumes
    a single step from the first commanded value to the last (the usage this module
    documents); a non-finite or degenerate (constant-commanded, zero step) input
    returns `None` where a time cannot be meaningfully defined rather than raising.
    """
    times = np.asarray(times_s, dtype=float)
    command = np.asarray(commanded, dtype=float)
    response = np.asarray(measured, dtype=float)
    if times.size == 0 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(response)):
        return StepResponseMetrics(None, 0.0, None, float("nan"))
    initial = float(response[0])
    final_command = float(command[-1])
    step = final_command - initial
    if step == 0.0:
        final_measured = float(response[-1])
        return StepResponseMetrics(None, 0.0, None, final_measured - final_command)

    def _level(frac):
        return initial + frac * step

    def _first_crossing(level):
        reached = response >= level if step > 0.0 else response <= level
        indices = np.flatnonzero(reached)
        return None if indices.size == 0 else float(times[indices[0]])

    t10, t90 = _first_crossing(_level(0.1)), _first_crossing(_level(0.9))
    rise_time = None if (t10 is None or t90 is None) else max(0.0, t90 - t10)

    extreme = float(np.max(response)) if step > 0.0 else float(np.min(response))
    overshoot = max(0.0, (extreme - final_command) / step) if step > 0.0 else max(0.0, (final_command - extreme) / step)

    band = abs(settling_band_frac * step)
    outside = np.flatnonzero(np.abs(response - final_command) > band)
    settling_time = 0.0 if outside.size == 0 else (None if outside[-1] == times.size - 1 else float(times[outside[-1] + 1]))

    steady_state_error = float(response[-1]) - final_command
    return StepResponseMetrics(rise_time, overshoot, settling_time, steady_state_error)


def analyze_velocity_tracking(commanded: Sequence[float], measured: Sequence[float]) -> Dict[str, float]:
    """RMS and max absolute tracking error over a recorded (or synthetic) trial --
    pure arithmetic, same "not run here, unit-tested on synthetic data" status as
    `analyze_step_response`."""
    command = np.asarray(commanded, dtype=float)
    response = np.asarray(measured, dtype=float)
    error = response - command
    return {
        "rms_error": float(np.sqrt(np.mean(error ** 2))) if error.size else float("nan"),
        "max_abs_error": float(np.max(np.abs(error))) if error.size else float("nan"),
    }
