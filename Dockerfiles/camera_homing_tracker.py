#!/usr/bin/env python3
"""Steer the Gazebo Hummingbird with the camera-homing law.

Reads the two ROS cameras and PX4's own telemetry, and sends a PX4 offboard
NED velocity. The red-pixel test and the sightline are the homing law. This
module is a thin ROS/MAVLink shell: per-camera buffering, timestamp-aware
vehicle state, independent setpoint streaming, off-control-path recording and
comms watchdogs all live in importable `rotorpy.guidance` modules so they are
unit-testable without ROS (see `rotorpy/guidance/{observation_buffer,
vehicle_state,setpoint_stream,recording_worker,watchdog}.py`).

Gazebo's own model state is read only for recording and outcome scoring
(`_pose_truth`); the control loop -- homing law, startup hold and arming --
uses PX4's reported attitude and local position exclusively, the same as a
real vehicle would. `rotorpy/guidance/scoring.py` is the truth-side module that
combines vehicle and target truth; no guidance/estimator module imports it (see
`tests/test_homing_truth_boundary.py`).

Startup is the `startup.StartupGate` state machine (bounded service waits, arming
attempts, motor start, climb and stabilisation); a timeout is a STARTUP_FAILURE
outcome. Errors are explicit: `run_failure.RecoverableError` is counted against a
bounded budget, everything else is fatal -- the setpoint stream is latched and
`run_lifecycle.RunLifecycle` writes exactly one outcome (also on failed startup and on
shutdown), flushing and closing the recorder on every termination path. `PX4's `ATTITUDE_QUATERNION`/`LOCAL_POSITION_NED` time
stamps are PX4's own boot clock, not Gazebo's simulated clock that the camera
header stamps use; lacking a live flight to calibrate that offset, this
shell timestamps ingested telemetry with `rospy.get_time()` at arrival as an
approximation (documented here, not claimed as verified).
"""

import atexit
import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import rospy
from gazebo_msgs.srv import GetJointProperties, GetLinkProperties, GetModelState, SetLinkProperties
from pymavlink import mavutil
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

from rotorpy.frames import enu_to_ned
from rotorpy.guidance.gazebo_bridge import (
    attitude_gazebo_to_homing,
    attitude_homing_to_ned_frd,
    heading_from_homing_attitude,
    velocity_gazebo_to_ned,
)
from rotorpy.guidance.gazebo_record import GazeboHomingRecorder, gazebo_page_path
from rotorpy.guidance.homing import HomingCommand
from rotorpy.guidance.homing_config import (
    PROPOSED_REACH_RADIUS_M,
    PROPOSED_SERVICE_WAIT_TIMEOUT_S,
    PROPOSED_STARTUP_MODE,
    PROPOSED_TELEMETRY_TIMEOUT_S,
    alignment_from_config,
    guidance_from_config,
    load_homing_config,
    startup_thresholds,
    watchdog_timeouts,
)
from rotorpy.guidance.observation_buffer import ObservationBuffer, PROPOSED_CAMERA_FUSION_MAX_AGE_S, select_fresh_or_recent
from rotorpy.guidance.outcome import RunOutcome, classify_run
from rotorpy.guidance.recording_worker import PROPOSED_RECORDING_QUEUE_DEPTH, RecordingWorker
from rotorpy.guidance.run_failure import (
    EstimatorFailure,
    FatalError,
    OffboardFailure,
    RecoverableBudget,
    RecoverableError,
    ServiceUnavailable,
    SimulatorFailure,
    StartupFailure,
    failure_for_kind,
    failure_reason,
    is_recoverable,
    outcome_flags,
)
from rotorpy.guidance.run_lifecycle import OUTCOME_PREFIX, RunLifecycle
from rotorpy.guidance.scoring import TargetShape, TruthScorer
from rotorpy.guidance.setpoint_stream import PROPOSED_FAILURE_COMMAND, PROPOSED_SETPOINT_EXPIRY_S, SetpointStream
from rotorpy.guidance.startup import (
    StartupCommand,
    StartupConfig,
    StartupGate,
    StartupInputs,
    StartupMode,
    StartupPhase,
    bounded_service_wait,
    parse_startup_mode,
)
from rotorpy.guidance.step_decision import command_and_stamp
from rotorpy.guidance.telemetry import (
    detection_fields_from_command,
    frame_to_float,
    setpoint_fields_from_command,
    start_run_telemetry,
    target_motion_fields_from_command,
)
from rotorpy.guidance.tracking import TrackingState
from rotorpy.guidance.vehicle_state import VehicleState
from rotorpy.guidance.watchdog import CommsWatchdogs

_OFFBOARD_MAIN = 6
_BODY_LINKS = ("base_link", "/imu_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3", "gps0::link")
_ROTOR_JOINTS = ("rotor_0_joint", "rotor_1_joint", "rotor_2_joint", "rotor_3_joint")
# Gazebo slows rotor joints by 10; their armed idle speed is 10 rad/s.
_MOTOR_READY_SPEED = 15.0
_TARGET_MODEL = "homing_target"
# Proposed, not verified: bound on waiting for the first PX4 heartbeat.
_PROPOSED_HEARTBEAT_WAIT_S = 30.0
_GAZEBO_SERVICES = (
    "/gazebo/get_link_properties",
    "/gazebo/set_link_properties",
    "/gazebo/get_joint_properties",
    "/gazebo/unpause_physics",
    "/gazebo/get_model_state",
)
# Run outcomes that mean the infrastructure, not the homing law, ended the run.
_INFRASTRUCTURE_OUTCOMES = (
    RunOutcome.STARTUP_FAILURE,
    RunOutcome.SIMULATOR_FAILURE,
    RunOutcome.OFFBOARD_FAILURE,
    RunOutcome.ESTIMATOR_FAILURE,
)
# Proposed, not verified: how often the independent setpoint publisher ticks, decoupled
# from the camera/perception rate configured as "camera_rate_hz".
_PROPOSED_SETPOINT_RATE_HZ = 50.0
_POSITION_IGNORE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)
_ACCELERATION_IGNORE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
)


def _position_target_type_mask(enable_acceleration_feedforward):
    """The `SET_POSITION_TARGET_LOCAL_NED` type mask: position/yaw-rate always
    ignored, acceleration ignored unless `enable_acceleration_feedforward` -- the
    documented, off-by-default semantics flag (plan Phase 7;
    `homing_config.PROPOSED_ENABLE_ACCELERATION_FEEDFORWARD`, `HomingGuidance.
    enable_acceleration_feedforward`). Default `False` keeps today's mask (the
    AX/AY/AZ ignore bits stay set, same as every pre-Phase-7 flight): acceleration
    fields stay masked out unless this is explicitly, separately validated and
    flipped on -- never "the fix" by itself (project history: accel feedforward was
    unstable; see the plan's own Phase 7 ruling).
    """
    if enable_acceleration_feedforward:
        return _POSITION_IGNORE & ~_ACCELERATION_IGNORE
    return _POSITION_IGNORE


# Comms links whose loss latches the setpoint stream fatal. The image link is excluded on
# purpose: camera dropouts are recoverable and handled by command expiry.
_FATAL_WATCHDOG_LINKS = ("heartbeat", "telemetry")
_MAVLINK_POLL_TYPES = ("HEARTBEAT", "STATUSTEXT", "ATTITUDE_QUATERNION", "LOCAL_POSITION_NED")


def _placeholder_command():
    """A benign, unlocked zero command: lets setpoints keep streaming before PX4's first
    attitude/local-position sample arrives, without fabricating anything from Gazebo truth."""
    return HomingCommand(
        np.zeros(3), np.zeros(3), np.zeros(3), 0.0, False, False, None, None, None
    )


def _lazy(obj, attr_name, factory):
    """The value stored at `attr_name` on `obj`, built by `factory()` on first use (for
    trackers constructed via `__new__` in tests, which skip `__init__`)."""
    value = getattr(obj, attr_name, None)
    if value is None:
        value = factory()
        setattr(obj, attr_name, value)
    return value


def _lock_for(obj, attr_name):
    """The `threading.Lock` stored at `attr_name` on `obj`, creating it on first use.

    Lazy so tests that construct a tracker via `__new__` (bypassing `__init__`, as most of
    `tests/test_gazebo_homing_startup.py` does to exercise one method at a time) do not
    need to also set up every lock `__init__` would have created.
    """
    lock = getattr(obj, attr_name, None)
    if lock is None:
        lock = threading.Lock()
        setattr(obj, attr_name, lock)
    return lock


class CameraHomingTracker(object):
    def __init__(self, config, model_name, page_path=None, startup_mode=PROPOSED_STARTUP_MODE):
        self.model_name = model_name
        # Explicit, validated before anything else is touched: a typo must not silently
        # pick a mode. "airborne_spawn" is the legacy simulation mode (gravity off until the
        # rotors spin); "takeoff" is the preferred path (see `startup.py`).
        self.startup_mode = parse_startup_mode(startup_mode)
        # Taken from the vehicle's own heading at the first PX4 attitude sample, never
        # from the target.
        self._startup_yaw = None
        self._startup_height = -float(config["start_ned"][2])
        self.guidance = guidance_from_config(config)
        self.page_path = page_path
        # Truth for scoring the outcome and recording the flight path only. It never
        # reaches guidance, the vehicle-state estimate, or the startup/arming logic.
        self._target_height_m = float(config["gazebo_target_height_m"])
        self._scorer = TruthScorer(
            TargetShape(float(config["gazebo_target_radius_m"]), self._target_height_m),
            timeout_s=float(config["timeout_s"]),
            reach_radius_m=PROPOSED_REACH_RADIUS_M,
            ground_down_m=-float(config["ground_z_m"]),
            static_target_ground_ned=config["target_ground_ned"],
        )
        self.telemetry = None
        if page_path is not None:
            self.telemetry = start_run_telemetry(Path(page_path).parent / "gazebo_runs", config)
        self._page_printed = False
        self.recorder = None if page_path is None else GazeboHomingRecorder(config)
        self._recording = RecordingWorker(self._record_sink, maxsize=PROPOSED_RECORDING_QUEUE_DEPTH)
        self._lifecycle = RunLifecycle(self._write_outcome_record, closers=self._closers())
        self._budget = RecoverableBudget()
        self._startup_error = None
        self._observations = ObservationBuffer(cameras=("forward", "upward"))
        self._vehicle_state = VehicleState(alignment_from_config(config, self.startup_mode))
        self._watchdogs = CommsWatchdogs(watchdog_timeouts())
        self._setpoints = SetpointStream(
            publish=self._publish_setpoint,
            expiry_s=PROPOSED_SETPOINT_EXPIRY_S,
            # Explicit, documented decision (not an unexamined default): send nothing on
            # expiry/fatal, so PX4 loses its offboard stream and falls back to its own
            # configured failsafe (RTL/land/hold) instead of this tracker fabricating a
            # velocity command of its own.
            failure_command=PROPOSED_FAILURE_COMMAND,
            now=time.monotonic,
        )
        # Protects cross-thread pymavlink access (not thread-safe) between the step
        # thread (`_poll_mavlink`, `_engage`) and the independent setpoint-publisher
        # thread (`_send`).
        self._mav_lock = threading.Lock()
        # Protects `_startup_done`/`_gravity_on`/`_startup_yaw`/`_last_transmitted`,
        # written by the step thread and/or the setpoint-publisher thread.
        self._flags_lock = threading.Lock()
        # Guards `_shutdown_done` atomically (minor a, fix round 1): `shutdown()` can be
        # called concurrently from rospy's shutdown hook, `atexit`, and the main loop's
        # `finally`; a bare `getattr`/`setattr` flag can race and run the body twice.
        self._shutdown_lock = threading.Lock()
        self._shutdown_done = False
        # The most recently queued ("outcome", ...) recording item not yet confirmed
        # processed (minor a, fix round 1): if `_recording.close(timeout)` returns while
        # the worker thread is still draining it, `shutdown()` can write this one
        # directly instead of a wrong SIMULATOR_FAILURE shadowing a real, decided
        # outcome. Cleared once `_record_sink` actually processes it.
        self._pending_outcome_item = None
        self._command = None
        self._sent = 0
        self._pending_status = None
        self._outcome_queued = False
        self._guidance_stamp = None
        self._last_transmitted = None
        self._offboard = False
        self._last_status = None
        self._hb_armed = False
        self._hb_mode = None
        self._gravity_on = True
        self._startup_done = False
        self._gate = StartupGate(
            StartupConfig(mode=self.startup_mode, hold_height_m=self._startup_height, **startup_thresholds()),
            now_s=time.monotonic(),
        )
        # Startup problems (a missing service, no heartbeat) must still end the run with an
        # outcome, so they are held and reported by the first `step()` instead of raised
        # out of the constructor where nothing could record them.
        try:
            self._connect()
        except Exception as exc:
            self._startup_error = exc if isinstance(exc, StartupFailure) else StartupFailure(
                "init_failed:{}: {}".format(type(exc).__name__, exc)
            )
            self._setpoints.mark_fatal("startup_failure: {}".format(self._startup_error.reason))

    def _connect(self):
        """The bounded part of startup that needs ROS/Gazebo/PX4 to be up."""
        bounded_service_wait(
            _GAZEBO_SERVICES,
            lambda name, remaining: rospy.wait_for_service(name, timeout=remaining),
            PROPOSED_SERVICE_WAIT_TIMEOUT_S,
            time.monotonic,
        )
        self._get_link = rospy.ServiceProxy("/gazebo/get_link_properties", GetLinkProperties)
        self._set_link = rospy.ServiceProxy("/gazebo/set_link_properties", SetLinkProperties)
        self._get_joint = rospy.ServiceProxy("/gazebo/get_joint_properties", GetJointProperties)
        if self.startup_mode is StartupMode.AIRBORNE_SPAWN:
            # The estimator is not ready at spawn. Hold the 30 m start until offboard is flying.
            self._set_gravity(False)
        rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
        rospy.Subscriber("/homing/forward/image_raw", Image, self._on_forward, queue_size=1)
        rospy.Subscriber("/homing/upward/image_raw", Image, self._on_upward, queue_size=1)
        self._model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self._mav = mavutil.mavlink_connection("udpin:127.0.0.1:14540", source_system=255)
        print("waiting for PX4 heartbeat on 14540", flush=True)
        if self._mav.wait_heartbeat(timeout=_PROPOSED_HEARTBEAT_WAIT_S) is None:
            raise StartupFailure("heartbeat_timeout")
        self._watchdogs.touch("heartbeat", time.monotonic())
        print("PX4 system {} (startup mode {})".format(self._mav.target_system, self.startup_mode.value), flush=True)
        # Independent of the camera-rate step() loop: a stale command must stop being sent
        # even if perception or the main loop stalls.
        rospy.Timer(rospy.Duration(1.0 / _PROPOSED_SETPOINT_RATE_HZ), self._on_setpoint_timer)

    # --- lightweight receivers -----------------------------------------------------

    def _on_forward(self, msg):
        self._on_image(msg, camera="forward")

    def _on_upward(self, msg):
        self._on_image(msg, camera="upward")

    def _on_image(self, msg, camera):
        self._observations.offer(
            camera,
            capture_s=_stamp_seconds(msg),
            receipt_s=time.monotonic(),
            raw_bytes=msg.data,
            width=msg.width,
            height=msg.height,
            encoding=msg.encoding,
            step=msg.step,
        )
        # Touch only after a successful decode: a corrupt/garbage frame (bad encoding,
        # stride or buffer size) must not make the image link look healthy.
        self._watchdogs.touch("image", time.monotonic())

    def _poll_mavlink(self):
        """Drain whatever PX4 messages have arrived; never blocks waiting for one."""
        while True:
            # pymavlink is not thread-safe: serialise against the setpoint-publisher
            # thread's `_send`, which also touches `self._mav`.
            with _lock_for(self, "_mav_lock"):
                msg = self._mav.recv_match(type=list(_MAVLINK_POLL_TYPES), blocking=False)
            if msg is None:
                return
            kind = msg.get_type()
            wall_now = time.monotonic()
            if kind == "STATUSTEXT":
                if not self._offboard and msg.text != self._last_status:
                    self._last_status = msg.text
                    print("px4: {}".format(msg.text), flush=True)
            elif kind == "HEARTBEAT":
                if msg.type == mavutil.mavlink.MAV_TYPE_GCS:
                    continue
                self._watchdogs.touch("heartbeat", wall_now)
                self._hb_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
                self._hb_mode = msg.custom_mode
            elif kind == "ATTITUDE_QUATERNION":
                self._watchdogs.touch("telemetry", wall_now)
                # PX4 quaternion fields are scalar-first (w, x, y, z); ours are scalar-last.
                quat_xyzw = np.array([msg.q2, msg.q3, msg.q4, msg.q1], dtype=float)
                self._vehicle_state.ingest_attitude(
                    sim_s=rospy.get_time(), wall_s=wall_now, quat_estimator_xyzw=quat_xyzw
                )
            elif kind == "LOCAL_POSITION_NED":
                self._watchdogs.touch("telemetry", wall_now)
                self._vehicle_state.ingest_local_position(
                    wall_s=wall_now,
                    position_estimator_ned=[msg.x, msg.y, msg.z],
                    velocity_estimator_ned=[msg.vx, msg.vy, msg.vz],
                )

    def _pose_truth(self):
        """Gazebo ground truth: recording and outcome scoring only, never guidance."""
        state = self._model_state(self.model_name, "")
        if not state.success:
            raise ServiceUnavailable(state.status_message)
        pose = state.pose
        twist = state.twist
        position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        velocity = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        return position, attitude_gazebo_to_homing(quat), velocity_gazebo_to_ned(velocity)

    def _target_truth(self):
        """Truth: the target's ground-contact point (NED) now, or None when it cannot be
        read (the scorer then uses the configured static point). Scoring only."""
        model_state = getattr(self, "_model_state", None)
        if model_state is None:
            return None
        try:
            state = model_state(_TARGET_MODEL, "")
        except Exception:
            return None
        if not state.success:
            return None
        centre = enu_to_ned(np.array([state.pose.position.x, state.pose.position.y, state.pose.position.z], dtype=float))
        # The model pose is the cylinder's centre; the ground contact is half its height below.
        return centre + np.array([0.0, 0.0, self._target_height_m / 2.0])

    def _flight_state(self):
        return (
            self._hb_armed
            and self._hb_mode is not None
            and ((int(self._hb_mode) >> 16) & 0xFF) == _OFFBOARD_MAIN
            and not self._watchdogs.stale("heartbeat", time.monotonic())
        )

    def step(self):
        """One control tick. Returns the run-ending status or None; raises on failure.

        `RecoverableError`s that escape are bounded by `handle_step_error`; a
        `FatalError` (or any unexpected exception) ends the run.
        """
        error = getattr(self, "_startup_error", None)
        if error is not None:
            raise error
        self._tick_recoverable = False
        status = self._step()
        if not self._tick_recoverable:
            _lazy(self, "_budget", RecoverableBudget).ok()
        return status

    def _note_recoverable(self, exc):
        """Count a recoverable error this tick absorbed; raises the fatal escalation once the
        bounded budget is spent."""
        self._tick_recoverable = True
        escalation = _lazy(self, "_budget", RecoverableBudget).note(exc)
        print("recoverable error: {}".format(exc), file=sys.stderr, flush=True)
        if escalation is not None:
            raise escalation

    def _gate_for(self, wall_now):
        gate = getattr(self, "_gate", None)
        if gate is None:
            gate = StartupGate(
                StartupConfig(
                    mode=getattr(self, "startup_mode", parse_startup_mode(PROPOSED_STARTUP_MODE)),
                    hold_height_m=self._startup_height, **startup_thresholds(),
                ),
                now_s=wall_now,
            )
            self._gate = gate
        return gate

    def _step(self):
        wall_now = time.monotonic()
        self._poll_mavlink()
        self._offboard = self._flight_state()
        # Every stale link is reported in telemetry (`stale_links`). Once homing has begun, a
        # *lost* telemetry or heartbeat link (alive once, now silent) is fatal and latches the
        # stream (decision, proposed default: a PX4 link blip ends the run; the reason is
        # recorded in the stats) -- and the run ends with an outcome, it is not left
        # streaming nothing. Before homing the loss is not latched: the startup gate drops
        # back to waiting for a fresh estimate / confirmed offboard, and its own bounds turn
        # a link that never returns into a STARTUP_FAILURE. The image watchdog is
        # deliberately never fatal: a camera dropout self-heals through command expiry and
        # resumes on the next fresh frame. A never-seen link is startup, not loss.
        self._stale_links = sorted(self._watchdogs.any_stale(wall_now))
        lost_fatal = sorted(name for name in self._watchdogs.lost(wall_now) if name in _FATAL_WATCHDOG_LINKS)
        if lost_fatal and self._startup_done:
            reason = "watchdog_lost:" + ",".join(lost_fatal)
            self._setpoints.mark_fatal(reason)
            raise (OffboardFailure if "heartbeat" in lost_fatal else EstimatorFailure)(reason)
        forward_obs, forward_fresh = self._observations.take_fresh("forward")
        upward_obs, upward_fresh = self._observations.take_fresh("upward")
        fresh = forward_fresh or upward_fresh
        capture_sim_s = None
        if forward_fresh and forward_obs is not None:
            capture_sim_s = forward_obs.capture_s
        elif upward_fresh and upward_obs is not None:
            capture_sim_s = upward_obs.capture_s
        sample = self._vehicle_state.sample_at(capture_sim_s, wall_now)
        # Only this step thread consumes the discontinuity (the publisher thread's own
        # `sample_at` calls cannot swallow it).
        if self._vehicle_state.take_discontinuity() is not None:
            # The sim clock reset/jumped/paused: the previous sightline/timestamp the
            # guidance law holds is no longer meaningfully adjacent in time to the next
            # one it will see.
            self.guidance.reset_derivative_history()
        # The homing law's attitude convention (`attitude_gazebo_to_homing`'s output); it
        # is what every `guidance.update` call consumes and what a replay must feed back.
        homing_attitude = (
            None if sample.attitude_quat_world is None else attitude_homing_to_ned_frd(sample.attitude_quat_world)
        )
        if self._startup_yaw is None and homing_attitude is not None:
            with _lock_for(self, "_flags_lock"):
                self._startup_yaw = heading_from_homing_attitude(homing_attitude)
        updated = fresh or self._command is None
        have_estimate = (
            sample
            and homing_attitude is not None
            and sample.velocity_ned is not None
            and not self._watchdogs.stale("telemetry", wall_now)
            # Minor (c), fix round 1: the combined "telemetry" watchdog is touched by
            # *either* ATTITUDE_QUATERNION or LOCAL_POSITION_NED, so it can stay fresh
            # while only one of the two keeps arriving. Check each `VehicleSample` age
            # on its own too, so a stalled attitude or position stream is caught even
            # while the other keeps the combined watchdog touched.
            and sample.attitude_age_s is not None
            and sample.attitude_age_s <= PROPOSED_TELEMETRY_TIMEOUT_S
            and sample.position_age_s is not None
            and sample.position_age_s <= PROPOSED_TELEMETRY_TIMEOUT_S
        )
        # A fresh frame from one camera must not make the *other* camera's stale cached
        # frame appear fresh, but that other camera's still-valid recent observation
        # (within the fusion max age) is still usable context, not unconditionally `None`.
        forward_used = select_fresh_or_recent(forward_obs, forward_fresh, wall_now, PROPOSED_CAMERA_FUSION_MAX_AGE_S)
        upward_used = select_fresh_or_recent(upward_obs, upward_fresh, wall_now, PROPOSED_CAMERA_FUSION_MAX_AGE_S)
        forward_image = None if forward_used is None else frame_to_float(forward_used.image)
        upward_image = None if upward_used is None else frame_to_float(upward_used.image)
        receipts = [obs.receipt_s for obs, is_fresh in ((forward_obs, forward_fresh), (upward_obs, upward_fresh))
                    if is_fresh and obs is not None]
        observation_wall_s = max(receipts) if receipts else wall_now
        # Advanced before the command is computed: the gate decision (including whether
        # homing opens this very tick) does not depend on `command`, and running it first
        # lets `compute_command` below see an up-to-date `self._startup_done` -- so
        # guidance starts on the same tick the gate opens, not one tick late.
        self._advance_startup(wall_now, have_estimate, sample, homing_attitude)

        def compute_command():
            # Finding 3 (fix round 1): guidance is not fed anything until the startup
            # gate has opened. Its tracking state machine is a one-way latch (`ABORT`
            # never recovers) and its loss clocks only advance on `update()` calls, so
            # calling it during a long takeoff climb -- target out of view, nothing to
            # do with it yet -- could latch a spurious `ABORT`/`TARGET_LOST` from pure
            # startup, or leave it stuck in `ABORT` once homing opens. The placeholder
            # keeps its `STARTUP` state untouched until there is an actual homing tick.
            if not self._startup_done:
                return _placeholder_command()
            # `t` is the image's own *capture* time (when the frame -- and the
            # `sample` above, already drawn from the T3 `VehicleState` history at
            # this same instant -- actually describe the world), not the processing
            # instant: every detection/dt/estimator-update computation inside
            # `guidance.update` must line up with the vehicle state it was given, not
            # with whenever this tick happens to run. `control_t` is the separate,
            # later processing instant the resulting command actually takes effect;
            # only `TargetMotionEstimator.predict_to` (reached through `_estimator`)
            # uses it, to compensate for exactly the gap between those two times.
            # Falls back to `rospy.get_time()` only before any frame has ever
            # arrived (`capture_sim_s is None`), matching the pre-Phase-7 behavior.
            measurement_t = capture_sim_s if capture_sim_s is not None else rospy.get_time()
            return self.guidance.update(
                measurement_t, sample.height_above_ground_m, homing_attitude, sample.velocity_ned,
                forward_image, upward_image, control_t=rospy.get_time(),
            )

        command, stamp_wall_s = command_and_stamp(
            have_estimate=have_estimate, fresh=fresh, previous_command=self._command,
            placeholder_command=_placeholder_command(), compute_command=compute_command,
            observation_wall_s=observation_wall_s,
        )
        self._command = command
        self._sent = getattr(self, "_sent", 0) + 1
        # `stamp_wall_s` is `None` on the hold-last-command branch: do NOT restamp the
        # setpoint stream there, or its expiry never fires and a dead camera/dead PX4
        # estimate never stops the stream (the held time must keep aging).
        if stamp_wall_s is not None:
            self._guidance_stamp = stamp_wall_s
        if self._startup_done:
            if stamp_wall_s is not None:
                self._setpoints.set_command(command, stamp_wall_s)
            elif getattr(self, "_guidance_stamp", None) is not None:
                # No fresh frame the very tick homing opened (rare): hand over the
                # guidance command at its own age once one exists, so a command computed
                # long ago expires instead of looking fresh.
                self._setpoints.set_command(command, self._guidance_stamp)
        status = None
        # `self._startup_done` here: a command computed *before* the gate opened was
        # unconditionally the placeholder (never fed to guidance above), so it cannot
        # itself be an `ABORT`. Discarding it until homing is active also protects a
        # resumed tracker whose startup history is unknown.
        if self._startup_done and command is not None and getattr(command, "tracking_state", None) is TrackingState.ABORT:
            # The explicit, one-way failure response: do NOT keep issuing normal
            # stale-command homing after `ABORT` (decision, proposed default: map it to
            # the T3 `SetpointStream` failure-response path and the T1 `TARGET_LOST`
            # outcome, not a silently-held last command).
            abort_reason = getattr(command, "abort_reason", None) or "unspecified"
            self._setpoints.mark_fatal("target_lost:" + abort_reason)
            status = "target_lost"
        if command is not None:
            truth = self._read_truth() if self._startup_done else None
            if truth is not None:
                # Termination comes from the truth scorer alone: it needs neither the
                # recorder nor any camera frame.
                status = status or self._score_truth(truth)
            status = status or getattr(self, "_pending_status", None)
            if truth is not None and self.recorder is not None:
                forward_picture = forward_obs.image if (fresh and forward_obs is not None) else None
                upward_picture = upward_obs.image if (fresh and upward_obs is not None) else None
                self.recorder.observe(
                    rospy.get_time(), truth["position_enu"], truth["attitude"], forward_picture, upward_picture,
                    command, target_ground_ned=truth["target_ground_ned"],
                )
            if self.telemetry is not None:
                self._recording.submit((
                    "step", self._sent, wall_now, sample, homing_attitude, command, updated,
                    forward_obs.image if (updated and forward_obs is not None) else None,
                    upward_obs.image if (updated and upward_obs is not None) else None,
                    {
                        "forward": None if forward_obs is None else forward_obs.capture_s,
                        "upward": None if upward_obs is None else upward_obs.capture_s,
                    },
                    {
                        "forward": None if forward_obs is None else forward_obs.receipt_s,
                        "upward": None if upward_obs is None else upward_obs.receipt_s,
                    },
                ))
            if status is not None:
                self._submit_outcome(status)
            print(
                "locked={} tracking={} source={} v=[{:.1f} {:.1f} {:.1f}]".format(
                    command.locked,
                    command.tracking,
                    command.source,
                    command.velocity_ned[0],
                    command.velocity_ned[1],
                    command.velocity_ned[2],
                ),
                flush=True,
            )
        return status

    # --- startup ---------------------------------------------------------------------

    def _advance_startup(self, wall_now, have_estimate, sample, homing_attitude):
        """Run the startup gate for this tick and act on it. Returns True on the tick homing opens.

        Streams the startup setpoint (restamped every step) while the gate returns one --
        that depends on the PX4 estimate only, never on a camera image -- and makes gate
        failures fatal. The homing command is not transmitted while the gate is open.
        """
        gate = self._gate_for(wall_now)
        if self._startup_done and gate.phase is not StartupPhase.HOMING:
            # A tracker resumed straight into homing (no gate history): nothing to gate.
            return False
        motors_ready = None
        if gate.phase is StartupPhase.MOTOR_START:
            try:
                motors_ready = bool(self._motors_ready())
            except RecoverableError as exc:
                self._note_recoverable(exc)
                motors_ready = False
        decision = gate.update(StartupInputs(
            now_s=wall_now,
            estimate_fresh=bool(have_estimate),
            attitude_valid=homing_attitude is not None,
            height_m=sample.height_above_ground_m,
            velocity_ned=sample.velocity_ned,
            # Minor (c), fix round 1: separate data -- `_hb_armed` is PX4's own raw ARM
            # state; `_offboard` (via `_flight_state`) is "armed AND in offboard mode".
            # Feeding the same combined flag into both fields hid the raw ARM state from
            # the gate (it only ever reads `armed and offboard`, so this changes no
            # decision today, but a future consumer of `armed` alone would otherwise see
            # the wrong thing). Falls back to `_offboard` itself when a test-constructed
            # tracker never set `_hb_armed` (no heartbeat ever polled).
            armed=bool(getattr(self, "_hb_armed", self._offboard)),
            offboard=bool(self._offboard),
            motors_ready=motors_ready,
        ))
        if decision.failure is not None:
            raise failure_for_kind(decision.failure_kind, decision.failure)
        if decision.gravity_enabled is not None and decision.gravity_enabled != self._gravity_on:
            self._set_gravity(decision.gravity_enabled)
        if decision.request_arm:
            self._engage()
        if decision.homing_active:
            if self._startup_done:
                return False
            with _lock_for(self, "_flags_lock"):
                self._startup_done = True
            print("startup gate open; beginning camera homing", flush=True)
            return True
        if decision.setpoint_velocity_ned is not None and self._startup_yaw is not None:
            self._setpoints.set_command(
                StartupCommand(velocity_ned=decision.setpoint_velocity_ned, yaw_rad=self._startup_yaw), wall_now
            )
        return False

    # --- truth side: scoring and termination -------------------------------------------

    def _read_truth(self):
        """Gazebo truth for scoring/recording, or None when unreadable this tick (a
        recoverable, bounded failure). Never feeds guidance."""
        try:
            position_enu, attitude, velocity = self._pose_truth()
        except RecoverableError as exc:
            self._note_recoverable(exc)
            return None
        return {
            "t": rospy.get_time(),
            "position_enu": position_enu,
            "attitude": attitude,
            "velocity": velocity,
            "target_ground_ned": self._target_truth(),
        }

    def _score_truth(self, truth):
        """Feed the truth scorer; returns its termination status (`ground`/`timeout`/`target`)."""
        scorer = self._scorer
        if truth["target_ground_ned"] is not None:
            scorer.add_target_sample(truth["t"], truth["target_ground_ned"])
        return scorer.add_vehicle_sample(truth["t"], enu_to_ned(np.asarray(truth["position_enu"], dtype=float)))

    def on_collision(self, kind):
        """Hook for a simulator collision event (`"target"` or `"ground"`).

        Not subscribed to a live Gazebo contact topic here (no simulator available): the
        scorer's segment tests already catch contact between samples; this is the second,
        independent source once such a topic is configured.
        """
        status = self._scorer.note_collision(rospy.get_time(), kind)
        if status is not None:
            # The next `step()` reports it (and queues the one outcome) like any other end.
            self._pending_status = status
        return status

    # --- outcomes ---------------------------------------------------------------------

    def _submit_recording_outcome(self, item):
        """Queue an `("outcome", ...)` recording item, remembering it (minor a, fix
        round 1) so `shutdown()` can still write it directly if the worker thread has
        not drained it by the time the recording queue is closed."""
        self._pending_outcome_item = item
        self._recording.submit(item)

    def _submit_outcome(self, status):
        """Queue the run's one outcome (after the pending tick records, so the log is ordered)."""
        if getattr(self, "_outcome_queued", False):
            return
        self._outcome_queued = True
        scorer = self._scorer
        outcome = scorer.outcome(target_lost=status == "target_lost")
        if outcome is None:
            outcome = {
                "ground": RunOutcome.GROUND_MISS, "timeout": RunOutcome.TIMEOUT,
                "target": RunOutcome.TARGET_REACHED, "target_lost": RunOutcome.TARGET_LOST,
            }.get(status, RunOutcome.SIMULATOR_FAILURE)
        result = scorer.result()
        self._submit_recording_outcome(("outcome", outcome, status, {
            "miss_m": scorer.miss_m(),
            "target_contact": result.target_contact,
            "ground_contact": result.ground_contact,
            "min_distance_time_s": result.min_distance_time_s,
            # Finding 2 (fix round 1): the ground-projected miss and whether the target
            # truth was fully available, alongside the closest-surface-approach reported
            # above, so a replay/report can show both without recomputing them.
            "ground_miss_m": result.ground_miss_m,
            "target_truth_unavailable": result.target_truth_unavailable,
            "scored_samples": result.matched_samples,
            "unmatched_samples": result.unmatched_samples,
        }))

    def _lifecycle_for(self):
        lifecycle = getattr(self, "_lifecycle", None)
        if lifecycle is None:
            lifecycle = RunLifecycle(self._write_outcome_record, closers=self._closers())
            self._lifecycle = lifecycle
        return lifecycle

    def _write_outcome_record(self, outcome, reason, fields):
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return False
        return bool(telemetry.finish(outcome, reason=reason, **fields))

    def _closers(self):
        return [("telemetry", self._close_telemetry)]

    def _close_telemetry(self):
        telemetry = getattr(self, "telemetry", None)
        if telemetry is not None:
            # Frames held for a pre-event window are written too: a failed run is exactly
            # when they are wanted.
            telemetry.close(flush_frames=True)

    def exit_code(self):
        """0 when homing ran to a flight outcome, 1 when infrastructure ended the run."""
        outcome = self._lifecycle_for().outcome
        return 1 if outcome is None or outcome in _INFRASTRUCTURE_OUTCOMES else 0

    def handle_step_error(self, exc):
        """What to do with an exception `step()` raised. True: recoverable, keep stepping.
        False: the run is over -- the stream is latched and exactly one outcome is queued."""
        if is_recoverable(exc):
            escalation = _lazy(self, "_budget", RecoverableBudget).note(exc)
            if escalation is None:
                return True
            exc = escalation
        self.note_failure(exc)
        reason = failure_reason(exc)
        self._setpoints.mark_fatal(
            "{}_failure: {}".format(exc.kind, reason) if isinstance(exc, FatalError) else "step_exception: {}".format(exc)
        )
        if not getattr(self, "_outcome_queued", False):
            self._outcome_queued = True
            outcome = classify_run(**outcome_flags(exc))
            self._submit_recording_outcome(("outcome", outcome, reason, {"error": type(exc).__name__}))
        return False

    # --- recording: off the control path, a slow sink cannot stall setpoints -------------

    def _record_sink(self, item):
        kind = item[0]
        if kind == "step":
            self._record_step(*item[1:])
        elif kind == "outcome":
            if item is getattr(self, "_pending_outcome_item", None):
                self._pending_outcome_item = None
            self._lifecycle_for().finish(item[1], item[2], **item[3])

    def _record_step(
        self, seq, wall_now, sample, homing_attitude, command, updated, forward_image, upward_image,
        image_capture_s, image_receipt_s,
    ):
        """Hand one tick to telemetry. Recording errors never reach the control loop."""
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        try:
            now = rospy.get_time()
            gate = getattr(self, "_gate", None)
            if self._startup_done:
                state = "HOMING"
            else:
                state = "STARTUP" if gate is None else gate.phase.value
            stats = dict(self._setpoints.stats_snapshot(), stale_links=list(getattr(self, "_stale_links", ())))
            with _lock_for(self, "_flags_lock"):
                transmitted = getattr(self, "_last_transmitted", None)
            telemetry.step(
                seq=seq,
                t_s=now,
                height_m=None if sample.height_above_ground_m is None else float(sample.height_above_ground_m),
                # The homing-law attitude convention: what `guidance.update` actually
                # consumed this tick, so a replay reconstructs the identical call.
                attitude_quat=homing_attitude,
                velocity_ned=sample.velocity_ned,
                forward=forward_image,
                upward=upward_image,
                updated=updated,
                command=command,
                image_capture_s=image_capture_s,
                image_receipt_s=image_receipt_s,
                attitude_age_s=sample.attitude_age_s,
                position_age_s=sample.position_age_s,
                velocity_age_s=sample.velocity_age_s,
                processing_start_s=now,
                processing_end_s=rospy.get_time(),
                position_ned=sample.position_ned,
                measured_velocity_ned=sample.velocity_ned,
                transmitted_setpoint=transmitted,
                state=state,
                px4_mode=None if self._hb_mode is None else str(self._hb_mode),
                camera=command.source,
                failure_reason=stats.get("fatal_reason"),
                setpoint_stats=stats,
                **detection_fields_from_command(command),
                **target_motion_fields_from_command(command),
                **setpoint_fields_from_command(command),
            )
        except Exception as exc:
            print("telemetry step failed: {}".format(exc), file=sys.stderr, flush=True)

    def note_failure(self, exc):
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        try:
            telemetry.event("step_failed", reason=str(exc))
        except Exception:
            pass

    # --- independent periodic setpoint publishing -----------------------------------

    def _on_setpoint_timer(self, _event):
        self._setpoints.tick()

    def _publish_setpoint(self, command):
        """Send one offboard velocity setpoint. Called by the setpoint stream's own thread;
        the command is whatever the stream holds (startup setpoint or homing command)."""
        self._send(command)

    def _send(self, command):
        velocity = command.velocity_ned
        yaw = command.yaw_rad
        accel = command.acceleration_ned
        enable_feedforward = bool(
            getattr(getattr(self, "guidance", None), "enable_acceleration_feedforward", False)
        )
        type_mask = _position_target_type_mask(enable_feedforward)
        # pymavlink is not thread-safe: serialise against the step thread's
        # `_poll_mavlink`/`_engage`, which also touch `self._mav`.
        with _lock_for(self, "_mav_lock"):
            self._mav.mav.set_position_target_local_ned_send(
                0,
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                type_mask,
                0, 0, 0,
                float(velocity[0]), float(velocity[1]), float(velocity[2]),
                float(accel[0]), float(accel[1]), float(accel[2]),
                float(yaw), 0.0,
            )
        with _lock_for(self, "_flags_lock"):
            self._last_transmitted = [float(velocity[0]), float(velocity[1]), float(velocity[2])]

    def _motors_ready(self):
        for name in _ROTOR_JOINTS:
            joint = self._get_joint("{}::{}".format(self.model_name, name))
            if not joint.success:
                raise ServiceUnavailable(joint.status_message)
            if not joint.rate:
                raise ServiceUnavailable("Gazebo returned no velocity for {}".format(name))
            if not np.isfinite(joint.rate[0]) or abs(joint.rate[0]) < _MOTOR_READY_SPEED:
                return False
        return True

    def _engage(self):
        target = (self._mav.target_system, self._mav.target_component)
        # pymavlink is not thread-safe: serialise against the setpoint-publisher
        # thread's `_send`, which also touches `self._mav`.
        with _lock_for(self, "_mav_lock"):
            self._mav.mav.command_long_send(
                target[0], target[1],
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                float(_OFFBOARD_MAIN), 0, 0, 0, 0, 0,
            )
            self._mav.mav.command_long_send(
                target[0], target[1],
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
                1, 0, 0, 0, 0, 0, 0,
            )
        print(
            "offboard and arm requested (armed={} custom_mode={})".format(self._hb_armed, self._hb_mode),
            flush=True,
        )

    def _set_gravity(self, enabled):
        for name in _BODY_LINKS:
            link = "{}::{}".format(self.model_name, name)
            props = self._get_link(link)
            if not props.success:
                raise RuntimeError(props.status_message)
            result = self._set_link(
                link, props.com, bool(enabled), props.mass,
                props.ixx, props.ixy, props.ixz, props.iyy, props.iyz, props.izz,
            )
            if not result.success:
                raise RuntimeError(result.status_message)
        with _lock_for(self, "_flags_lock"):
            self._gravity_on = bool(enabled)
        print("gravity {}".format("on" if enabled else "off"), flush=True)

    def write_page(self):
        if self.recorder is None or self._page_printed:
            return None
        path = self.recorder.write(self.page_path)
        if path is None:
            return None
        self._page_printed = True
        print(path, flush=True)
        return path

    def shutdown(self):
        """Drain the recording queue, then settle the outcome and close every sink. Safe to
        call from every termination path (orderly, failure, rospy shutdown, atexit): a run
        torn down with no outcome gets one infrastructure outcome rather than silence."""
        with _lock_for(self, "_shutdown_lock"):
            # Atomic check-and-set (minor a, fix round 1): `rospy.on_shutdown`, `atexit`
            # and the main loop's own `finally` can all reach this concurrently; a bare
            # `getattr`/`setattr` flag can race two callers into the body at once.
            if getattr(self, "_shutdown_done", False):
                return
            self._shutdown_done = True
        self._recording.close(timeout=5.0)
        lifecycle = self._lifecycle_for()
        pending = getattr(self, "_pending_outcome_item", None)
        if lifecycle.outcome is None and pending is not None:
            # The worker thread raced the close timeout with a real, already-decided
            # outcome still in its queue (minor a, fix round 1): write that one directly
            # instead of letting `ensure_outcome` shadow it with a wrong
            # SIMULATOR_FAILURE "shutdown_before_outcome".
            self._record_sink(pending)
        lifecycle.ensure_outcome("shutdown_before_outcome")
        lifecycle.close()

    def mark_fatal(self, reason=None):
        """A fatal `step()` failure stops the normal setpoint command (one-way latch)
        instead of streaming whatever was last held forever."""
        self._setpoints.mark_fatal(reason)


def _stamp_seconds(msg):
    try:
        return float(msg.header.stamp.to_sec())
    except Exception:
        return None


def _announce_init_failure(page_path, exc):
    """`main()`'s construction block (config load, the truth scorer, telemetry, the
    recorder, ...) ran before any run directory or outcome sink existed. A failure there
    used to be a bare traceback with no `RUN_OUTCOME` line -- "every run has an outcome"
    failed on exactly this path. Announce the one outcome on stdout (the same line
    `RunLifecycle.finish` would print) and, best effort, drop a minimal outcome JSON file
    next to the configured page/results path. Never raises."""
    failure = exc if isinstance(exc, FatalError) else StartupFailure(
        "init_failed:{}: {}".format(type(exc).__name__, exc)
    )
    outcome = classify_run(**outcome_flags(failure))
    fields = {"outcome": outcome.value, "reason": failure_reason(failure), "persisted": False}
    try:
        print("{} {}".format(OUTCOME_PREFIX, json.dumps(fields, default=str)), flush=True)
    except Exception:
        pass
    if page_path is None:
        return
    try:
        outcome_path = Path(page_path).with_suffix(".outcome.json")
        outcome_path.parent.mkdir(parents=True, exist_ok=True)
        outcome_path.write_text(json.dumps(fields, default=str))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Camera-homing tracker for the Gazebo Hummingbird.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="hummingbird_0")
    parser.add_argument(
        "--startup-mode",
        choices=[mode.value for mode in StartupMode],
        default=os.environ.get("HOMING_STARTUP_MODE", PROPOSED_STARTUP_MODE),
        help="takeoff: spawn on the ground, arm and climb (preferred). airborne_spawn: legacy "
        "simulation mode, spawned at the start height with gravity disabled until the rotors spin.",
    )
    args = parser.parse_args()
    # Bounded fallback (finding 1, fix round 1): config load, the truth scorer, telemetry
    # and the recorder all run here, outside the tracker's own `_connect` try/except --
    # a failure must still end with exactly one announced/recorded outcome, not a
    # traceback and silence.
    page_path = None
    try:
        page_path = gazebo_page_path(args.config)
        config = load_homing_config(args.config)
        rospy.init_node("camera_homing_tracker", anonymous=True)
        tracker = CameraHomingTracker(config, args.model, page_path, startup_mode=args.startup_mode)
    except Exception as exc:
        _announce_init_failure(page_path, exc)
        return 1
    rospy.on_shutdown(tracker.write_page)
    rospy.on_shutdown(tracker.shutdown)
    atexit.register(tracker.write_page)
    atexit.register(tracker.shutdown)
    rate = rospy.Rate(float(config["camera_rate_hz"]))
    status = None
    try:
        while not rospy.is_shutdown():
            try:
                status = tracker.step()
            except Exception as exc:
                print("tracker step failed: {}".format(exc), flush=True)
                # Recoverable errors are counted against a bounded budget and the loop goes
                # on. Anything else is fatal: the handler latches the setpoint stream and
                # queues the run's one outcome, and the run ends here (no catch-and-continue).
                if tracker.handle_step_error(exc):
                    rate.sleep()
                    continue
                status = "failed"
                break
            if status is not None:
                break
            rate.sleep()
    finally:
        # Every exit path -- orderly, failed, interrupted -- saves the page and then drains
        # the recording queue, writes the outcome and closes the recorder.
        path = tracker.write_page()
        if status is not None:
            print("flight finished: {}".format(status), flush=True)
            if path is None:
                print("flight finished without camera frames; no recording was written", file=sys.stderr, flush=True)
        tracker.shutdown()
    exit_code = getattr(tracker, "exit_code", None)
    return 0 if exit_code is None else exit_code()


if __name__ == "__main__":
    sys.exit(main())
