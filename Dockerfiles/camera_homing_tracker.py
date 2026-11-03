#!/usr/bin/env python3
"""Steer the Gazebo Hummingbird with the camera-homing law.

Reads the two ROS cameras and the Gazebo pose, and sends a PX4 offboard
NED velocity. The red-pixel test and the sightline are the homing law.
"""

import atexit
import argparse
import sys
import threading
from pathlib import Path

import numpy as np
import rospy
from gazebo_msgs.srv import GetJointProperties, GetLinkProperties, GetModelState, SetLinkProperties
from pymavlink import mavutil
from sensor_msgs.msg import Image
from std_srvs.srv import Empty

from rotorpy.frames import enu_to_ned
from rotorpy.guidance.gazebo_bridge import attitude_gazebo_to_homing, velocity_gazebo_to_ned
from rotorpy.guidance.gazebo_record import GazeboHomingRecorder, gazebo_page_path
from rotorpy.guidance.homing_config import (
    guidance_from_config,
    initial_velocity_from_config,
    load_homing_config,
)
from rotorpy.guidance.outcome import classify_run
from rotorpy.guidance.telemetry import frame_to_float, start_run_telemetry

_OFFBOARD_MAIN = 6
_BODY_LINKS = ("base_link", "/imu_link", "rotor_0", "rotor_1", "rotor_2", "rotor_3", "gps0::link")
_ROTOR_JOINTS = ("rotor_0_joint", "rotor_1_joint", "rotor_2_joint", "rotor_3_joint")
_STARTUP_CLIMB_SPEED = 3.0
# Gazebo slows rotor joints by 10; their armed idle speed is 10 rad/s.
_MOTOR_READY_SPEED = 15.0
_MOTOR_START_TIMEOUT = 15.0
_STARTUP_HOLD_TIMEOUT = 20.0
_POSITION_IGNORE = (
    mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_Z_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE
    | mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE
)


def _image_rgb(msg):
    channels = 4 if msg.encoding in ("rgba8", "bgra8") else 3
    frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    if msg.encoding.startswith("bgr"):
        frame = frame[:, :, [2, 1, 0]]
    return np.array(frame[:, :, :3], dtype=np.uint8)


class CameraHomingTracker(object):
    _forward_raw = None
    _upward_raw = None
    _image_times = {}

    def __init__(self, config, model_name, page_path=None):
        self.model_name = model_name
        initial_velocity = initial_velocity_from_config(config)
        self._startup_yaw = float(np.arctan2(initial_velocity[1], initial_velocity[0]))
        self._startup_height = -float(config["start_ned"][2])
        self.guidance = guidance_from_config(config)
        self.page_path = page_path
        # Truth for scoring the outcome only. It never reaches guidance.
        self._target_ground_ned = np.array(config["target_ground_ned"], dtype=float)
        self.telemetry = None
        if page_path is not None:
            self.telemetry = start_run_telemetry(Path(page_path).parent / "gazebo_runs", config)
        self._forward_raw = None
        self._upward_raw = None
        self._image_times = {}
        self._page_printed = False
        self.recorder = None if page_path is None else GazeboHomingRecorder(config)
        self._lock = threading.Lock()
        self._forward = None
        self._upward = None
        self._fresh = False
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
        print("PX4 system {}".format(self._mav.target_system), flush=True)

    def _on_forward(self, msg):
        self._store(msg, forward=True)

    def _on_upward(self, msg):
        self._store(msg, forward=False)

    def _store(self, msg, forward):
        raw = _image_rgb(msg)
        image = frame_to_float(raw)
        camera = "forward" if forward else "upward"
        times = (_stamp_seconds(msg), rospy.get_time())
        with self._lock:
            if forward:
                self._forward, self._forward_raw = image, raw
            else:
                self._upward, self._upward_raw = image, raw
            self._image_times = dict(self._image_times, **{camera: times})
            self._fresh = True

    def _pose(self):
        state = self._model_state(self.model_name, "")
        if not state.success:
            raise RuntimeError(state.status_message)
        pose = state.pose
        twist = state.twist
        position = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=float)
        quat = np.array([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w])
        velocity = np.array([twist.linear.x, twist.linear.y, twist.linear.z])
        return position, attitude_gazebo_to_homing(quat), velocity_gazebo_to_ned(velocity)

    def step(self):
        with self._lock:
            fresh = self._fresh
            forward = None if self._forward is None else self._forward
            upward = None if self._upward is None else self._upward
            raw = (self._forward_raw, self._upward_raw)
            image_times = dict(self._image_times)
            self._fresh = False
        process_start = rospy.get_time()
        position, attitude, velocity = self._pose()
        height = float(position[2])
        updated = fresh or self._command is None
        if fresh:
            command = self.guidance.update(rospy.get_time(), height, attitude, velocity, forward, upward)
        elif self._command is None:
            command = self.guidance.update(rospy.get_time(), height, attitude, velocity, None, None)
            raw = (None, None)
        else:
            command = self._command
        self._command = command
        self._send(command, height)
        self._sent += 1
        self._offboard = self._flight_state()
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
        if self._offboard and self._gravity_on and not self._startup_done:
            self._update_startup_hold(rospy.get_time(), height, velocity[2])
        status = None
        if self.recorder is not None and self._startup_done:
            pictures = (forward, upward) if fresh else (None, None)
            status = self.recorder.observe(
                rospy.get_time(), position, attitude, pictures[0], pictures[1], command
            )
        self._record_step(
            process_start, position, attitude, velocity, command, updated, raw, image_times,
        )
        if status is not None:
            self._record_outcome(status, position)
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

    def _record_step(self, start, position, attitude, velocity, command, updated, raw, image_times):
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
                height_m=float(position[2]),
                attitude_quat=attitude,
                velocity_ned=velocity,
                forward=raw[0] if updated else None,
                upward=raw[1] if updated else None,
                updated=updated,
                command=command,
                image_capture_s={cam: t[0] for cam, t in image_times.items()},
                image_receipt_s={cam: t[1] for cam, t in image_times.items()},
                processing_start_s=start,
                processing_end_s=now,
                position_ned=enu_to_ned(np.asarray(position, dtype=float)),
                measured_velocity_ned=velocity,
                transmitted_setpoint=sent if self._startup_done else None,
                state="HOMING" if self._startup_done else "STARTUP",
                px4_mode=None if self._hb_mode is None else str(self._hb_mode),
                camera=command.source,
            )
        except Exception as exc:
            print("telemetry step failed: {}".format(exc), file=sys.stderr, flush=True)

    def _record_outcome(self, status, position):
        telemetry = getattr(self, "telemetry", None)
        if telemetry is None:
            return
        try:
            miss = float(np.linalg.norm(enu_to_ned(np.asarray(position, dtype=float)) - self._target_ground_ned))
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

    def _send(self, command, height):
        if self._startup_done:
            velocity = command.velocity_ned
        elif self._gravity_on:
            velocity = (0.0, 0.0, float(np.clip(height - self._startup_height, -2.0, 2.0)))
        else:
            velocity = (0.0, 0.0, -_STARTUP_CLIMB_SPEED)
        accel = command.acceleration_ned
        self._mav.mav.set_position_target_local_ned_send(
            0,
            self._mav.target_system,
            self._mav.target_component,
            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
            _POSITION_IGNORE,
            0, 0, 0,
            float(velocity[0]), float(velocity[1]), float(velocity[2]),
            float(accel[0]), float(accel[1]), float(accel[2]),
            float(command.yaw_rad if self._startup_done else self._startup_yaw), 0.0,
        )

    def _update_startup_hold(self, now, height, velocity_down):
        if abs(height - self._startup_height) <= 1.0 and abs(velocity_down) <= 0.5:
            if self._settled_since is None:
                self._settled_since = now
            elif now - self._settled_since >= 1.0:
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

    def _flight_state(self):
        while True:
            msg = self._mav.recv_match(type=["HEARTBEAT", "STATUSTEXT"], blocking=False)
            if msg is None:
                break
            if msg.get_type() == "STATUSTEXT":
                if not self._offboard and msg.text != self._last_status:
                    self._last_status = msg.text
                    print("px4: {}".format(msg.text), flush=True)
                continue
            if msg.type == mavutil.mavlink.MAV_TYPE_GCS:
                continue
            self._hb_armed = bool(msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED)
            self._hb_mode = msg.custom_mode
        return (
            self._hb_armed
            and self._hb_mode is not None
            and ((int(self._hb_mode) >> 16) & 0xFF) == _OFFBOARD_MAIN
        )


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
    atexit.register(tracker.write_page)
    rate = rospy.Rate(float(config["camera_rate_hz"]))
    while not rospy.is_shutdown():
        try:
            status = tracker.step()
        except Exception as exc:
            print("tracker step failed: {}".format(exc), flush=True)
            note = getattr(tracker, "note_failure", None)
            if note is not None:
                note(exc)
            rate.sleep()
            continue
        if status is not None:
            print("flight finished: {}".format(status), flush=True)
            if tracker.write_page() is None:
                print("flight finished without camera frames; no recording was written", file=sys.stderr, flush=True)
            break
        rate.sleep()


if __name__ == "__main__":
    main()
