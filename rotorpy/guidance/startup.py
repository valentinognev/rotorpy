"""Startup gate: the explicit state machine between "PX4 is up" and "homing begins".

Pure and ROS-free. The tracker feeds `StartupGate.update` one `StartupInputs` snapshot per
step and acts on the returned `StartupDecision`; nothing here touches Gazebo truth.

Homing begins only when every one of these holds for `settle_hold_s` of continuous time:
a fresh PX4 estimate, a valid attitude, a finite height and velocity at the hold height
with bounded horizontal and vertical speed, and a confirmed armed + offboard state.

What is sent (or not) during startup -- an explicit, tested policy, independent of when
the first camera image arrives:

=============  =========================================================================
phase          setpoint stream
=============  =========================================================================
WAIT_ESTIMATE  nothing (no fresh PX4 estimate -> no setpoint; PX4 sees no stream)
ENGAGE         `startup_velocity` hold from the moment the estimate is fresh; the arm +
               offboard request is made only after `pre_engage_stream_s` of streaming
MOTOR_START    airborne-spawn only; the gravity-off climb hold
CLIMB          takeoff only; clipped climb toward the hold height
STABILIZE      clipped hold toward the hold height
HOMING         nothing from the gate: the homing command is transmitted
FAILED         nothing (terminal); the run ends with a STARTUP_FAILURE outcome
=============  =========================================================================

The tracker restamps the startup setpoint every step while the gate returns one, so the
stream stays alive on estimate freshness alone and goes silent (command expiry) the moment
the estimate stops being fresh.

Modes: `TAKEOFF` (preferred; spawn on the ground, arm, climb to the hold height) and
`AIRBORNE_SPAWN` (legacy simulation mode; spawn at the hold height with gravity disabled
until the rotors spin). The airborne mode is an explicit flag, never an implicit default.
"""

from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from rotorpy.guidance.run_failure import StartupFailure

__all__ = [
    "StartupCommand", "StartupConfig", "StartupDecision", "StartupFailure", "StartupGate", "StartupInputs",
    "StartupMode", "StartupPhase", "bounded_service_wait", "parse_startup_mode", "startup_velocity",
]

_HOLD_CLIP_M_S = 2.0


class StartupMode(str, Enum):
    TAKEOFF = "takeoff"
    AIRBORNE_SPAWN = "airborne_spawn"


class StartupPhase(str, Enum):
    WAIT_ESTIMATE = "WAIT_ESTIMATE"
    ENGAGE = "ENGAGE"
    MOTOR_START = "MOTOR_START"
    CLIMB = "CLIMB"
    STABILIZE = "STABILIZE"
    HOMING = "HOMING"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StartupCommand:
    """The setpoint streamed while the gate is open: a velocity and the vehicle's own heading.

    Quacks like a `HomingCommand` for the setpoint stream's publisher (`velocity_ned`,
    `acceleration_ned`, `yaw_rad`). It never carries a target coordinate.
    """

    velocity_ned: Tuple[float, float, float]
    yaw_rad: float
    acceleration_ned: Tuple[float, float, float] = (0.0, 0.0, 0.0)


def parse_startup_mode(value) -> StartupMode:
    try:
        return StartupMode(value)
    except ValueError:
        raise ValueError(
            "unknown startup mode {!r}; expected one of {}".format(value, [mode.value for mode in StartupMode])
        )


@dataclass(frozen=True)
class StartupConfig:
    hold_height_m: float
    estimate_timeout_s: float
    pre_engage_stream_s: float
    arm_retry_period_s: float
    max_arm_attempts: int
    engage_confirm_timeout_s: float
    motor_start_timeout_s: float
    climb_timeout_s: float
    stabilize_timeout_s: float
    total_timeout_s: float
    settle_hold_s: float
    height_tolerance_m: float
    vertical_speed_tolerance_m_s: float
    horizontal_speed_tolerance_m_s: float
    climb_speed_m_s: float
    offboard_loss_grace_s: float
    mode: StartupMode = StartupMode.AIRBORNE_SPAWN

    def __post_init__(self) -> None:
        for name in (
            "estimate_timeout_s", "arm_retry_period_s", "engage_confirm_timeout_s",
            "motor_start_timeout_s", "climb_timeout_s", "stabilize_timeout_s", "total_timeout_s",
            "settle_hold_s", "height_tolerance_m", "vertical_speed_tolerance_m_s",
            "horizontal_speed_tolerance_m_s", "climb_speed_m_s", "offboard_loss_grace_s",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError("{} must be a positive finite number, got {!r}".format(name, value))
        if self.pre_engage_stream_s < 0.0 or not math.isfinite(self.pre_engage_stream_s):
            raise ValueError("pre_engage_stream_s must be a non-negative finite number")
        if int(self.max_arm_attempts) < 1:
            raise ValueError("max_arm_attempts must be at least 1")
        if not math.isfinite(self.hold_height_m):
            raise ValueError("hold_height_m must be finite")
        object.__setattr__(self, "mode", parse_startup_mode(self.mode))


@dataclass(frozen=True)
class StartupInputs:
    """One step's snapshot. `now_s` is wall-monotonic time so a stalled sim still times out."""

    now_s: float
    estimate_fresh: bool
    attitude_valid: bool
    height_m: Optional[float]
    velocity_ned: Optional[Sequence[float]]
    armed: bool
    offboard: bool
    motors_ready: Optional[bool] = None


@dataclass(frozen=True)
class StartupDecision:
    phase: StartupPhase
    request_arm: bool = False
    # Desired gravity state in the airborne-spawn mode (None: never touch gravity).
    gravity_enabled: Optional[bool] = None
    setpoint_velocity_ned: Optional[Tuple[float, float, float]] = None
    homing_active: bool = False
    failure: Optional[str] = None
    failure_kind: Optional[str] = None
    attempts: int = 0


def startup_velocity(
    mode: StartupMode,
    phase: StartupPhase,
    gravity_enabled: Optional[bool],
    height_m: float,
    hold_height_m: float,
    climb_speed_m_s: float,
) -> Optional[Tuple[float, float, float]]:
    """The NED velocity to stream for `phase`, or None when the gate sends nothing."""
    if phase not in (StartupPhase.ENGAGE, StartupPhase.MOTOR_START, StartupPhase.CLIMB, StartupPhase.STABILIZE):
        return None
    error = float(height_m) - float(hold_height_m)
    if mode is StartupMode.AIRBORNE_SPAWN:
        if not gravity_enabled:
            return (0.0, 0.0, -float(climb_speed_m_s))
        return (0.0, 0.0, float(np.clip(error, -_HOLD_CLIP_M_S, _HOLD_CLIP_M_S)))
    if phase is StartupPhase.ENGAGE:
        return (0.0, 0.0, 0.0)
    return (0.0, 0.0, float(np.clip(error, -climb_speed_m_s, climb_speed_m_s)))


class StartupGate(object):
    def __init__(self, config: StartupConfig, now_s: float) -> None:
        self.config = config
        self._started_s = float(now_s)
        self.phase = StartupPhase.WAIT_ESTIMATE
        self._entered_s = float(now_s)
        self.attempts = 0
        self._last_attempt_s: Optional[float] = None
        self._settled_since: Optional[float] = None
        self._offboard_lost_since: Optional[float] = None
        self._gravity_enabled: Optional[bool] = (
            False if config.mode is StartupMode.AIRBORNE_SPAWN else None
        )
        self._failure: Optional[str] = None
        self._failure_kind: Optional[str] = None

    @property
    def failure(self) -> Optional[str]:
        return self._failure

    @property
    def failure_kind(self) -> Optional[str]:
        return self._failure_kind

    def _enter(self, phase: StartupPhase, now_s: float) -> None:
        self.phase = phase
        self._entered_s = now_s
        self._settled_since = None

    def _fail(self, reason: str, kind: str = "startup") -> None:
        self.phase = StartupPhase.FAILED
        self._failure = reason
        self._failure_kind = kind

    def _decision(self, request_arm: bool = False, setpoint=None, homing_active: bool = False) -> StartupDecision:
        return StartupDecision(
            phase=self.phase,
            request_arm=request_arm,
            gravity_enabled=self._gravity_enabled,
            setpoint_velocity_ned=setpoint,
            homing_active=homing_active,
            failure=self._failure,
            failure_kind=self._failure_kind,
            attempts=self.attempts,
        )

    @staticmethod
    def _ready(inputs: StartupInputs) -> bool:
        if not (inputs.estimate_fresh and inputs.attitude_valid):
            return False
        if inputs.height_m is None or not math.isfinite(inputs.height_m):
            return False
        velocity = inputs.velocity_ned
        return velocity is not None and len(velocity) == 3 and all(math.isfinite(v) for v in velocity)

    def _stream(self, inputs: StartupInputs) -> Optional[Tuple[float, float, float]]:
        return startup_velocity(
            self.config.mode, self.phase, self._gravity_enabled, inputs.height_m,
            self.config.hold_height_m, self.config.climb_speed_m_s,
        )

    def update(self, inputs: StartupInputs) -> StartupDecision:
        now = float(inputs.now_s)
        config = self.config
        if self.phase is StartupPhase.FAILED:
            return self._decision()
        if self.phase is StartupPhase.HOMING:
            return self._update_homing(inputs, now)
        if now - self._started_s > config.total_timeout_s:
            self._fail("startup_timeout")
            return self._decision()
        if not self._ready(inputs):
            if self.phase is not StartupPhase.WAIT_ESTIMATE:
                self._enter(StartupPhase.WAIT_ESTIMATE, now)
            if now - self._entered_s > config.estimate_timeout_s:
                self._fail("estimate_timeout")
            return self._decision()
        if self.phase is StartupPhase.WAIT_ESTIMATE:
            self._enter(StartupPhase.ENGAGE, now)
            return self._decision(setpoint=self._stream(inputs))
        confirmed = inputs.armed and inputs.offboard
        if self.phase is not StartupPhase.ENGAGE and not confirmed:
            if self._gravity_enabled:
                self._fail("offboard_lost_after_gravity_release")
                return self._decision()
            self._enter(StartupPhase.ENGAGE, now)
        if self.phase is StartupPhase.ENGAGE:
            return self._update_engage(inputs, now, confirmed)
        if self.phase is StartupPhase.MOTOR_START:
            return self._update_motor_start(inputs, now)
        if self.phase is StartupPhase.CLIMB:
            return self._update_climb(inputs, now)
        return self._update_stabilize(inputs, now)

    def _after_engage(self, now: float) -> None:
        if self.config.mode is StartupMode.AIRBORNE_SPAWN and not self._gravity_enabled:
            self._enter(StartupPhase.MOTOR_START, now)
        else:
            self._enter(StartupPhase.CLIMB, now)

    def _update_engage(self, inputs: StartupInputs, now: float, confirmed: bool) -> StartupDecision:
        config = self.config
        if confirmed:
            self._after_engage(now)
            return self._phase_decision(inputs, now)
        request = False
        if now - self._entered_s >= config.pre_engage_stream_s:
            due = self._last_attempt_s is None or now - self._last_attempt_s >= config.arm_retry_period_s
            if self.attempts < config.max_arm_attempts and due:
                self.attempts += 1
                self._last_attempt_s = now
                request = True
            elif (
                self.attempts >= config.max_arm_attempts
                and self._last_attempt_s is not None
                and now - self._last_attempt_s >= config.engage_confirm_timeout_s
            ):
                self._fail("offboard_not_confirmed after {} attempts".format(self.attempts))
                return self._decision()
        return self._decision(request_arm=request, setpoint=self._stream(inputs))

    def _phase_decision(self, inputs: StartupInputs, now: float) -> StartupDecision:
        return self._decision(setpoint=self._stream(inputs))

    def _update_motor_start(self, inputs: StartupInputs, now: float) -> StartupDecision:
        if inputs.motors_ready:
            self._gravity_enabled = True
            self._enter(StartupPhase.STABILIZE, now)
            return self._decision(setpoint=self._stream(inputs))
        if now - self._entered_s > self.config.motor_start_timeout_s:
            self._fail("motor_start_timeout")
            return self._decision()
        return self._decision(setpoint=self._stream(inputs))

    def _update_climb(self, inputs: StartupInputs, now: float) -> StartupDecision:
        if abs(inputs.height_m - self.config.hold_height_m) <= self.config.height_tolerance_m:
            self._enter(StartupPhase.STABILIZE, now)
        elif now - self._entered_s > self.config.climb_timeout_s:
            self._fail("climb_timeout")
            return self._decision()
        return self._decision(setpoint=self._stream(inputs))

    def _is_stable(self, inputs: StartupInputs) -> bool:
        config = self.config
        vx, vy, vz = inputs.velocity_ned
        return (
            abs(inputs.height_m - config.hold_height_m) <= config.height_tolerance_m
            and math.hypot(vx, vy) <= config.horizontal_speed_tolerance_m_s
            and abs(vz) <= config.vertical_speed_tolerance_m_s
        )

    def _update_stabilize(self, inputs: StartupInputs, now: float) -> StartupDecision:
        if self._is_stable(inputs):
            if self._settled_since is None:
                self._settled_since = now
            if now - self._settled_since >= self.config.settle_hold_s:
                self._enter(StartupPhase.HOMING, now)
                self._offboard_lost_since = None
                return self._decision(homing_active=True)
        else:
            self._settled_since = None
        if now - self._entered_s > self.config.stabilize_timeout_s:
            self._fail("stabilization_timeout")
            return self._decision()
        return self._decision(setpoint=self._stream(inputs))

    def _update_homing(self, inputs: StartupInputs, now: float) -> StartupDecision:
        if inputs.armed and inputs.offboard:
            self._offboard_lost_since = None
            return self._decision(homing_active=True)
        if self._offboard_lost_since is None:
            self._offboard_lost_since = now
        if now - self._offboard_lost_since > self.config.offboard_loss_grace_s:
            self._fail("offboard_lost", kind="offboard")
            return self._decision()
        return self._decision(homing_active=True)


def bounded_service_wait(
    names: Sequence[str],
    wait_one: Callable[[str, float], None],
    timeout_s: float,
    clock: Callable[[], float],
) -> None:
    """Wait for each named service within one shared time budget.

    `wait_one(name, remaining_s)` raises on timeout (for example `rospy.wait_for_service`
    with its `timeout`). The first missing service becomes a `StartupFailure`.
    """
    start = clock()
    for name in names:
        remaining = timeout_s - (clock() - start)
        if remaining <= 0.0:
            raise StartupFailure("service_timeout:{}".format(name))
        try:
            wait_one(name, remaining)
        except Exception as exc:
            raise StartupFailure("service_timeout:{} ({})".format(name, exc))
