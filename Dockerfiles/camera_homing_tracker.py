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
real vehicle would. `PX4's `ATTITUDE_QUATERNION`/`LOCAL_POSITION_NED` time
stamps are PX4's own boot clock, not Gazebo's simulated clock that the camera
header stamps use; lacking a live flight to calibrate that offset, this
shell timestamps ingested telemetry with `rospy.get_time()` at arrival as an
approximation (documented here, not claimed as verified).
"""

import atexit
import argparse
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
from rotorpy.guidance.homing_config import alignment_from_config, guidance_from_config, load_homing_config, watchdog_timeouts
from rotorpy.guidance.observation_buffer import ObservationBuffer, PROPOSED_CAMERA_FUSION_MAX_AGE_S, select_fresh_or_recent
from rotorpy.guidance.outcome import classify_run
from rotorpy.guidance.recording_worker import PROPOSED_RECORDING_QUEUE_DEPTH, RecordingWorker
from rotorpy.guidance.setpoint_stream import PROPOSED_FAILURE_COMMAND, PROPOSED_SETPOINT_EXPIRY_S, SetpointStream
from rotorpy.guidance.step_decision import command_and_stamp
from rotorpy.guidance.telemetry import frame_to_float, start_run_telemetry
from rotorpy.guidance.vehicle_state import VehicleState
from rotorpy.guidance.watchdog import CommsWatchdogs

_OFFBOARD_MAIN = 6
_BODY_LINKS = ("base_link", "/imu_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3", "gps0::link")
_ROTOR_JOINTS = ("rotor_0_joint", "rotor_1_joint", "rotor_2_joint", "rotor_3_joint")
_STARTUP_CLIMB_SPEED = 3.0
# Gazebo slows rotor joints by 10; their armed idle speed is 10 rad/s.
_MOTOR_READY_SPEED = 15.0
_MOTOR_START_TIMEOUT = 15.0
_STARTUP_HOLD_TIMEOUT = 20.0
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
    def __init__(self, config, model_name, page_path=None):
        self.model_name = model_name
        # Taken from the vehicle's own heading at the first PX4 attitude sample, never
        # from the target.
        self._startup_yaw = None
        self._startup_height = -float(config["start_ned"][2])
        self.guidance = guidance_from_config(config)
        self.page_path = page_path
        # Truth for scoring the outcome and recording the flight path only. It never
        # reaches guidance, the vehicle-state estimate, or the startup/arming logic.
        self._target_ground_ned = np.array(config["target_ground_ned"], dtype=float)
        self.telemetry = None
        if page_path is not None:
            self.telemetry = start_run_telemetry(Path(page_path).parent / "gazebo_runs", config)
        self._page_printed = False
        self.recorder = None if page_path is None else GazeboHomingRecorder(config)
        self._recording = RecordingWorker(self._record_sink, maxsize=PROPOSED_RECORDING_QUEUE_DEPTH)
        self._observations = ObservationBuffer(cameras=("forward", "upward"))
        self._vehicle_state = VehicleState(alignment_from_config(config))
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
        # Protects `_startup_done`/`_gravity_on`/`_startup_yaw`, written by the step
        # thread and read by the setpoint-publisher thread's `_send`.
        self._flags_lock = threading.Lock()
        self._command = None
        self._sent = 0
        self._offboard = False
        self._last_status = None
        self._hb_armed = False
        self._hb_mode = None
        self._gravity_on = True
        self._motor_start_time = None
        self._startup_done = False
        self._hold_start_time = None
        self._settled_since = None
        rospy.wait_for_service("/gazebo/get_link_properties")
        rospy.wait_for_service("/gazebo/set_link_properties")
        rospy.wait_for_service("/gazebo/get_joint_properties")
        self._get_link = rospy.ServiceProxy("/gazebo/get_link_properties", GetLinkProperties)
        self._set_link = rospy.ServiceProxy("/gazebo/set_link_properties", SetLinkProperties)
        self._get_joint = rospy.ServiceProxy("/gazebo/get_joint_properties", GetJointProperties)
        # The estimator is not ready at spawn. Hold the 30 m start until offboard is flying.
        self._set_gravity(False)
        rospy.wait_for_service("/gazebo/unpause_physics")
        rospy.ServiceProxy("/gazebo/unpause_physics", Empty)()
        rospy.Subscriber("/homing/forward/image_raw", Image, self._on_forward, queue_size=1)
        rospy.Subscriber("/homing/upward/image_raw", Image, self._on_upward, queue_size=1)
        rospy.wait_for_service("/gazebo/get_model_state")
        self._model_state = rospy.ServiceProxy("/gazebo/get_model_state", GetModelState)
        self._mav = mavutil.mavlink_connection("udpin:127.0.0.1:14540", source_system=255)
        print("waiting for PX4 heartbeat on 14540", flush=True)
        self._mav.wait_heartbeat()
        self._watchdogs.touch("heartbeat", time.monotonic())
        print("PX4 system {}".format(self._mav.target_system), flush=True)
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
            raise RuntimeError(state.status_message)
        pose = state.pose
        twist = state.twist
        position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        velocity = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        return position, attitude_gazebo_to_homing(quat), velocity_gazebo_to_ned(velocity)

    def _flight_state(self):
        return (
            self._hb_armed
            and self._hb_mode is not None
            and ((int(self._hb_mode) >> 16) & 0xFF) == _OFFBOARD_MAIN
            and not self._watchdogs.stale("heartbeat", time.monotonic())
        )

    def step(self):
        wall_now = time.monotonic()
        self._poll_mavlink()
        self._offboard = self._flight_state()
        # Every stale link is reported in telemetry (`stale_links`), but only a *lost*
        # telemetry/heartbeat link (alive once, now silent) is fatal: the stream stops and
        # stays stopped (decision, proposed default: a PX4 link blip is fatal-forever; the
        # reason is recorded in the stats). The image watchdog is deliberately not fatal --
        # a camera dropout self-heals through command expiry and resumes on the next
        # fresh frame. A never-seen link is startup, not loss.
        self._stale_links = sorted(self._watchdogs.any_stale(wall_now))
        lost_fatal = sorted(name for name in self._watchdogs.lost(wall_now) if name in _FATAL_WATCHDOG_LINKS)
        if lost_fatal:
            self._setpoints.mark_fatal("watchdog_lost:" + ",".join(lost_fatal))
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
            sample.height_above_ground_m is not None
            and homing_attitude is not None
            and sample.velocity_ned is not None
            and not self._watchdogs.stale("telemetry", wall_now)
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

        def compute_command():
            return self.guidance.update(
                rospy.get_time(), sample.height_above_ground_m, homing_attitude, sample.velocity_ned,
                forward_image, upward_image,
            )

        command, stamp_wall_s = command_and_stamp(
            have_estimate=have_estimate, fresh=fresh, previous_command=self._command,
            placeholder_command=_placeholder_command(), compute_command=compute_command,
            observation_wall_s=observation_wall_s,
        )
        self._command = command
        # `stamp_wall_s` is `None` on the hold-last-command branch: do NOT restamp the
        # setpoint stream there, or its expiry never fires and a dead camera/dead PX4
        # estimate never stops the stream (the held time must keep aging).
        if stamp_wall_s is not None:
            self._setpoints.set_command(command, stamp_wall_s)
        self._sent += 1
        # PX4 rejects the first arm while the estimator is still starting.
        # Keep streaming setpoints and ask again until the heartbeat is armed in offboard.
        if not self._offboard and self._sent >= 20 and self._sent % 5 == 0:
            self._engage()
        if self._offboard and not self._gravity_on:
            if self._motor_start_time is None:
                self._motor_start_time = rospy.get_time()
            if self._motors_ready():
                self._set_gravity(True)
                self._hold_start_time = rospy.get_time()
            elif rospy.get_time() - self._motor_start_time > _MOTOR_START_TIMEOUT:
                raise RuntimeError("PX4 motors did not reach startup speed; gravity remains disabled")
        elif not self._offboard:
            self._motor_start_time = None
            self._settled_since = None
        if self._offboard and self._gravity_on and not self._startup_done and have_estimate:
            self._update_startup_hold(rospy.get_time(), sample.height_above_ground_m, sample.velocity_ned[2])
        status = None
        if command is not None:
            need_truth = self.recorder is not None or self.telemetry is not None
            if need_truth:
                position_truth, attitude_truth, velocity_truth = self._pose_truth()
                if self.recorder is not None and self._startup_done:
                    forward_picture = forward_obs.image if (fresh and forward_obs is not None) else None
                    upward_picture = upward_obs.image if (fresh and upward_obs is not None) else None
                    status = self.recorder.observe(
                        rospy.get_time(), position_truth, attitude_truth, forward_picture, upward_picture, command
                    )
                self._recording.submit((
                    "step", wall_now, sample, homing_attitude, command, updated,
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
                    self._recording.submit(("outcome", status, position_truth))
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

    # --- recording: off the control path, a slow sink cannot stall setpoints -------

    def _record_sink(self, item):
        kind = item[0]
        if kind == "step":
            self._record_step(*item[1:])
        elif kind == "outcome":
            self._record_outcome(item[1], item[2])

    def _record_step(
        self, wall_now, sample, homing_attitude, command, updated, forward_image, upward_image,
        image_capture_s, image_receipt_s,
    ):
        """Hand one tick to telemetry. Recording errors never reach the control loop."""
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        try:
            now = rospy.get_time()
            sent = np.array(command.velocity_ned, dtype=float)
            telemetry.step(
                seq=self._sent,
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
                transmitted_setpoint=sent if self._startup_done else None,
                state="HOMING" if self._startup_done else "STARTUP",
                px4_mode=None if self._hb_mode is None else str(self._hb_mode),
                camera=command.source,
                setpoint_stats=dict(
                    self._setpoints.stats_snapshot(), stale_links=list(getattr(self, "_stale_links", ()))
                ),
            )
        except Exception as exc:
            print("telemetry step failed: {}".format(exc), file=sys.stderr, flush=True)

    def _record_outcome(self, status, position_truth):
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        try:
            miss = float(
                np.linalg.norm(enu_to_ned(np.asarray(position_truth, dtype=float)) - self._target_ground_ned)
            )
            outcome = classify_run(
                miss_m=miss, ground_contact=status == "ground", timed_out=status == "timeout"
            )
            telemetry.finish(outcome, reason=status, miss_m=miss)
        except Exception as exc:
            print("telemetry outcome failed: {}".format(exc), file=sys.stderr, flush=True)

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
        """Fetch the current PX4 height estimate and send one offboard velocity setpoint."""
        height = self._vehicle_state.sample_at(None, time.monotonic()).height_above_ground_m
        if height is None:
            # No PX4 estimate yet: hold level (zero vertical rate) rather than guess.
            height = self._startup_height
        self._send(command, height)

    def _send(self, command, height):
        # Read the cross-thread flags together, under the same lock `step()` writes them
        # with, so `_send` (running on the setpoint-publisher thread) never sees a
        # partially-updated combination of them.
        with _lock_for(self, "_flags_lock"):
            startup_done = self._startup_done
            gravity_on = self._gravity_on
            startup_yaw = self._startup_yaw
        if startup_done:
            velocity = command.velocity_ned
        elif gravity_on:
            velocity = (0.0, 0.0, float(np.clip(height - self._startup_height, -2.0, 2.0)))
        else:
            velocity = (0.0, 0.0, -_STARTUP_CLIMB_SPEED)
        yaw = command.yaw_rad if startup_done or startup_yaw is None else startup_yaw
        accel = command.acceleration_ned
        # pymavlink is not thread-safe: serialise against the step thread's
        # `_poll_mavlink`/`_engage`, which also touch `self._mav`.
        with _lock_for(self, "_mav_lock"):
            self._mav.mav.set_position_target_local_ned_send(
                0,
                self._mav.target_system,
                self._mav.target_component,
                mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                _POSITION_IGNORE,
                0, 0, 0,
                float(velocity[0]), float(velocity[1]), float(velocity[2]),
                float(accel[0]), float(accel[1]), float(accel[2]),
                float(yaw), 0.0,
            )

    def _update_startup_hold(self, now, height, velocity_down):
        if abs(height - self._startup_height) <= 1.0 and abs(velocity_down) <= 0.5:
            if self._settled_since is None:
                self._settled_since = now
            elif now - self._settled_since >= 1.0:
                with _lock_for(self, "_flags_lock"):
                    self._startup_done = True
                print("startup hover settled; beginning camera homing", flush=True)
                return
        else:
            self._settled_since = None
        if self._hold_start_time is not None and now - self._hold_start_time > _STARTUP_HOLD_TIMEOUT:
            raise RuntimeError("PX4 did not stabilize at the startup height; holding position")

    def _motors_ready(self):
        for name in _ROTOR_JOINTS:
            joint = self._get_joint("{}::{}".format(self.model_name, name))
            if not joint.success:
                raise RuntimeError(joint.status_message)
            if not joint.rate:
                raise RuntimeError("Gazebo returned no velocity for {}".format(name))
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
        self._recording.close(timeout=5.0)

    def mark_fatal(self, reason=None):
        """A fatal `step()` failure stops the normal setpoint command (one-way latch)
        instead of streaming whatever was last held forever."""
        self._setpoints.mark_fatal(reason)


def _stamp_seconds(msg):
    try:
        return float(msg.header.stamp.to_sec())
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Camera-homing tracker for the Gazebo Hummingbird.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", default="hummingbird_0")
    args = parser.parse_args()
    config = load_homing_config(args.config)
    rospy.init_node("camera_homing_tracker", anonymous=True)
    tracker = CameraHomingTracker(config, args.model, gazebo_page_path(args.config))
    rospy.on_shutdown(tracker.write_page)
    rospy.on_shutdown(tracker.shutdown)
    atexit.register(tracker.write_page)
    atexit.register(tracker.shutdown)
    rate = rospy.Rate(float(config["camera_rate_hz"]))
    while not rospy.is_shutdown():
        try:
            status = tracker.step()
        except Exception as exc:
            print("tracker step failed: {}".format(exc), flush=True)
            note = getattr(tracker, "note_failure", None)
            if note is not None:
                note(exc)
            mark_fatal = getattr(tracker, "mark_fatal", None)
            if mark_fatal is not None:
                mark_fatal("step_exception: {}".format(exc))
            rate.sleep()
            continue
        if status is not None:
            print("flight finished: {}".format(status), flush=True)
            if tracker.write_page() is None:
                print("flight finished without camera frames; no recording was written", file=sys.stderr, flush=True)
            break
        rate.sleep()
    tracker.shutdown()


if __name__ == "__main__":
    main()
