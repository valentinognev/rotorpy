"""Verify the Gazebo startup hold and PX4 heartbeat handling without ROS."""

import importlib.util
import json
from pathlib import Path
import sys
import time
from types import ModuleType, SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from rotorpy.guidance.homing_config import load_homing_config


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def tracker_module(monkeypatch):
    rospy = ModuleType("rospy")
    monkeypatch.setitem(sys.modules, "rospy", rospy)
    for package, module, names in (
        ("gazebo_msgs", "srv", ("GetJointProperties", "GetLinkProperties", "GetModelState", "SetLinkProperties")),
        ("sensor_msgs", "msg", ("Image",)),
        ("std_srvs", "srv", ("Empty",)),
    ):
        monkeypatch.setitem(sys.modules, package, ModuleType(package))
        child = ModuleType(f"{package}.{module}")
        for name in names:
            setattr(child, name, type(name, (), {}))
        monkeypatch.setitem(sys.modules, child.__name__, child)

    spec = importlib.util.spec_from_file_location(
        "gazebo_homing_tracker", ROOT / "Dockerfiles" / "camera_homing_tracker.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gravity_hold_covers_all_model_links(tracker_module):
    model = ET.parse(ROOT / "gazeboModels" / "hummingbird" / "hummingbird.sdf.jinja")
    links = {link.attrib["name"] for link in model.findall("./model/link")}
    links.update(f"{include.findtext('name')}::link" for include in model.findall("./model/include"))
    assert set(tracker_module._BODY_LINKS) == links

    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker.model_name = "hummingbird_0"
    props = SimpleNamespace(
        success=True, com=None, mass=1, ixx=1, ixy=0, ixz=0, iyy=1, iyz=0, izz=1
    )
    calls = []
    tracker._get_link = lambda name: props
    tracker._set_link = lambda *args: calls.append(args) or SimpleNamespace(success=True)
    for enabled in (False, True):
        tracker._set_gravity(enabled)
        assert {args[0] for args in calls} == {f"hummingbird_0::{link}" for link in links}
        assert all(args[2] is enabled for args in calls)
        assert tracker._gravity_on is enabled
        calls.clear()


def test_physics_unpauses_only_after_gravity_is_disabled(tracker_module, monkeypatch):
    events = []
    rospy = tracker_module.rospy
    monkeypatch.setattr(
        rospy, "wait_for_service", lambda name, timeout=None: events.append(("wait", name)), raising=False
    )
    monkeypatch.setattr(rospy, "Subscriber", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(rospy, "Timer", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(rospy, "Duration", lambda seconds: seconds, raising=False)

    def service(name, _type):
        return lambda *args: events.append(("call", name))

    monkeypatch.setattr(rospy, "ServiceProxy", service, raising=False)
    monkeypatch.setattr(
        tracker_module.CameraHomingTracker, "_set_gravity",
        lambda self, enabled: events.append(("gravity", enabled)),
    )
    mav = SimpleNamespace(target_system=1, wait_heartbeat=lambda timeout=None: object())
    monkeypatch.setattr(tracker_module.mavutil, "mavlink_connection", lambda *args, **kwargs: mav)
    config = load_homing_config(ROOT / "examples" / "camera_homing.jsonc")
    tracker = tracker_module.CameraHomingTracker(config, "hummingbird_0")
    assert events.index(("gravity", False)) < events.index(("call", "/gazebo/unpause_physics"))
    tracker.shutdown()

    launcher = (ROOT / "Dockerfiles" / "sitl_hummingbird_run.sh").read_text()
    assert launcher.index("rosservice call /gazebo/pause_physics") < launcher.index(
        'spawn_sdf --spawn-file="${TARGET_SDF}"'
    )


def test_last_heartbeat_state_persists_until_a_new_heartbeat(tracker_module):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._hb_armed = False
    tracker._hb_mode = None
    tracker._offboard = False
    tracker._last_status = None
    tracker._watchdogs = tracker_module.CommsWatchdogs(tracker_module.watchdog_timeouts())
    mavlink = tracker_module.mavutil.mavlink
    messages = [
        SimpleNamespace(
            get_type=lambda: "HEARTBEAT",
            type=0,
            base_mode=mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
            custom_mode=tracker_module._OFFBOARD_MAIN << 16,
        )
    ]
    tracker._mav = SimpleNamespace(recv_match=lambda **kwargs: messages.pop(0) if messages else None)
    tracker._poll_mavlink()
    assert tracker._flight_state()
    assert tracker._flight_state()
    messages.append(SimpleNamespace(get_type=lambda: "HEARTBEAT", type=0, base_mode=0, custom_mode=0))
    tracker._poll_mavlink()
    assert not tracker._flight_state()


def test_image_watchdog_is_touched_only_after_a_successful_decode(tracker_module):
    # Touching the watchdog before decoding means a corrupt/garbage frame (a bad
    # `encoding`/`step`/buffer size) would still mark the image link "healthy".
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._watchdogs = tracker_module.CommsWatchdogs(tracker_module.watchdog_timeouts())
    tracker._observations = tracker_module.ObservationBuffer(cameras=("forward", "upward"))
    bad_msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 1.0)),
        data=b"\x00" * 4, width=4, height=4, encoding="yuv422", step=16,
    )
    with pytest.raises(ValueError):
        tracker._on_forward(bad_msg)
    assert tracker._watchdogs.stale("image", time.monotonic())
    good_msg = SimpleNamespace(
        header=SimpleNamespace(stamp=SimpleNamespace(to_sec=lambda: 1.0)),
        data=bytes(4 * 4 * 3), width=4, height=4, encoding="rgb8", step=4 * 3,
    )
    tracker._on_forward(good_msg)
    assert not tracker._watchdogs.stale("image", time.monotonic())


def test_normal_takeoff_and_motor_ready_threshold_match_the_model(tracker_module):
    airframe = (ROOT / "Dockerfiles" / "airframes" / "1100_gazebo-classic_hummingbird").read_text()
    assert "param set-default COM_THROW_EN 0" in airframe
    model = ET.parse(ROOT / "gazeboModels" / "hummingbird" / "hummingbird.sdf.jinja")
    motors = model.findall("./model/plugin[@filename='libgazebo_motor_model.so']")
    channels = model.findall("./model/plugin[@name='mavlink_interface']/control_channels/channel")
    assert len(motors) == len(channels) == len(tracker_module._ROTOR_JOINTS)
    for motor, channel in zip(motors, channels):
        slowdown = float(motor.findtext("rotorVelocitySlowdownSim"))
        idle = float(channel.findtext("zero_position_armed")) / slowdown
        maximum = float(motor.findtext("maxRotVelocity")) / slowdown
        assert idle < tracker_module._MOTOR_READY_SPEED < maximum


def test_startup_waits_for_all_rotors_and_reports_invalid_feedback(tracker_module):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker.model_name = "hummingbird_0"
    rates = {name: -tracker_module._MOTOR_READY_SPEED for name in tracker_module._ROTOR_JOINTS}
    tracker._get_joint = lambda name: SimpleNamespace(
        success=True, rate=[rates[name.split("::")[-1]]]
    )
    assert tracker._motors_ready()
    rates["rotor_3_joint"] = tracker_module._MOTOR_READY_SPEED - 1
    assert not tracker._motors_ready()
    rates["rotor_3_joint"] = np.nan
    assert not tracker._motors_ready()
    tracker._get_joint = lambda name: SimpleNamespace(success=True, rate=[])
    with pytest.raises(RuntimeError, match="no velocity"):
        tracker._motors_ready()
    tracker._get_joint = lambda name: SimpleNamespace(success=False, status_message="joint missing")
    with pytest.raises(RuntimeError, match="joint missing"):
        tracker._motors_ready()


def test_send_transmits_exactly_the_command_it_is_given(tracker_module):
    # The startup/homing policy lives in `StartupGate` and `step()`; `_send` is only the wire.
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    sent = []
    tracker._mav = SimpleNamespace(
        target_system=1, target_component=1,
        mav=SimpleNamespace(set_position_target_local_ned_send=lambda *args: sent.append(args)),
    )
    startup = tracker_module.StartupCommand(velocity_ned=(0.0, 0.0, -3.0), yaw_rad=0.3)
    tracker._send(startup)
    assert sent[-1][8:11] == (0.0, 0.0, -3.0)
    assert sent[-1][14] == 0.3
    assert tracker._last_transmitted == [0.0, 0.0, -3.0]
    command = SimpleNamespace(velocity_ned=[10.0, 5.0, 3.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.8)
    tracker._send(command)
    assert sent[-1][8:11] == (10.0, 5.0, 3.0)
    assert sent[-1][14] == 0.8
    assert tracker._last_transmitted == [10.0, 5.0, 3.0]


def test_send_masks_acceleration_by_default_but_not_when_feedforward_is_enabled(tracker_module):
    """Plan Phase 7: acceleration fields stay masked (ignored) in the mavlink setpoint
    unless `HomingGuidance.enable_acceleration_feedforward` (off by default) says
    otherwise -- the type mask toggles, not "unmasking acceleration" unconditionally."""
    assert tracker_module._position_target_type_mask(False) == tracker_module._POSITION_IGNORE
    with_feedforward = tracker_module._position_target_type_mask(True)
    assert with_feedforward != tracker_module._POSITION_IGNORE
    accel_bits = tracker_module._ACCELERATION_IGNORE
    assert with_feedforward & accel_bits == 0
    # Position/yaw-rate stay ignored either way -- only acceleration toggles.
    assert with_feedforward | accel_bits == tracker_module._POSITION_IGNORE

    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    command = SimpleNamespace(velocity_ned=[1.0, 0.0, 0.0], acceleration_ned=[0.1, 0.2, 0.3], yaw_rad=0.0)
    sent = []
    tracker._mav = SimpleNamespace(
        target_system=1, target_component=1,
        mav=SimpleNamespace(set_position_target_local_ned_send=lambda *args: sent.append(args)),
    )

    tracker.guidance = SimpleNamespace(enable_acceleration_feedforward=False)
    tracker._send(command)
    assert sent[-1][4] == tracker_module._POSITION_IGNORE

    tracker.guidance = SimpleNamespace(enable_acceleration_feedforward=True)
    tracker._send(command)
    assert sent[-1][4] == with_feedforward

    # No `guidance` attribute at all (matches every other `_send`-only test in this
    # file, built via `__new__`): defaults to masked, same as before this task.
    del tracker.guidance
    tracker._send(command)
    assert sent[-1][4] == tracker_module._POSITION_IGNORE


def test_mav_access_is_serialised_across_threads(tracker_module):
    # pymavlink is not thread-safe: the 50 Hz setpoint-publisher thread's `_send` and the
    # step thread's `_poll_mavlink` must never run inside the same connection at once.
    import threading

    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._hb_armed = False
    tracker._hb_mode = None
    tracker._last_status = None
    tracker._watchdogs = tracker_module.CommsWatchdogs(tracker_module.watchdog_timeouts())

    active = [0]
    overlap = []
    section_lock = threading.Lock()

    def enter():
        with section_lock:
            active[0] += 1
            if active[0] > 1:
                overlap.append(True)

    def leave():
        with section_lock:
            active[0] -= 1

    def slow_send(*args):
        enter()
        time.sleep(0.02)
        leave()

    def slow_recv(**kwargs):
        enter()
        time.sleep(0.02)
        leave()
        return None

    tracker._mav = SimpleNamespace(
        target_system=1, target_component=1,
        mav=SimpleNamespace(set_position_target_local_ned_send=slow_send),
        recv_match=slow_recv,
    )
    command = SimpleNamespace(velocity_ned=[1.0, 0.0, 0.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.0)
    send_thread = threading.Thread(target=lambda: tracker._send(command))
    poll_thread = threading.Thread(target=tracker._poll_mavlink)
    send_thread.start()
    poll_thread.start()
    send_thread.join()
    poll_thread.join()
    assert overlap == []


def _scorer(timeout_s=20.0):
    from rotorpy.guidance.scoring import TargetShape, TruthScorer

    return TruthScorer(
        TargetShape(radius_m=0.75, height_m=3.0), timeout_s=timeout_s,
        static_target_ground_ned=[100.0, 50.0, 0.0],
    )


def _new_tracker(tracker_module, monkeypatch, armed=True):
    """A `CameraHomingTracker` with its ROS/Gazebo/MAVLink I/O stubbed out, but real
    `rotorpy.guidance` collaborators (observation buffer, vehicle state, watchdogs,
    setpoint stream), so tests exercise the actual wiring `step()` relies on."""
    config = load_homing_config(ROOT / "examples" / "camera_homing.jsonc")
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._mav = SimpleNamespace(recv_match=lambda **kwargs: None)
    tracker._observations = tracker_module.ObservationBuffer(cameras=("forward", "upward"))
    tracker._vehicle_state = tracker_module.VehicleState(tracker_module.alignment_from_config(config))
    tracker._watchdogs = tracker_module.CommsWatchdogs(tracker_module.watchdog_timeouts())
    tracker._setpoints = tracker_module.SetpointStream(publish=lambda command: None, expiry_s=10.0)
    tracker._recording = SimpleNamespace(submit=lambda item: tracker._record_sink(item))
    tracker._command = None
    tracker._startup_yaw = 0.0
    tracker._startup_height = 30.0
    tracker._flight_state = lambda: armed
    tracker._gravity_on = False
    tracker._startup_done = False
    tracker._hb_mode = None
    tracker._scorer = _scorer()
    tracker.guidance = _FakeGuidance()
    tracker.recorder = None
    tracker.telemetry = None
    tracker._pose_truth = lambda: (np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    tracker._set_gravity = lambda enabled: setattr(tracker, "_gravity_on", enabled)
    monkeypatch.setattr(tracker_module.rospy, "get_time", lambda: 10.0, raising=False)
    return tracker


def _fresh_estimate(tracker, wall_s=None):
    """Ingest one PX4 estimate at rest at the 30 m airborne spawn (the configured estimator
    origin), and touch telemetry."""
    wall_s = time.monotonic() if wall_s is None else wall_s
    tracker._vehicle_state.ingest_attitude(sim_s=wall_s, wall_s=wall_s, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=wall_s, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    tracker._watchdogs.touch("telemetry", wall_s)


def test_gravity_stays_off_until_armed_offboard_motors_spin(tracker_module, monkeypatch):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    _fresh_estimate(tracker)
    tracker.step()  # estimate fresh -> ENGAGE
    tracker.step()  # armed + offboard confirmed -> MOTOR_START
    tracker.step()
    assert not tracker._gravity_on
    tracker._motors_ready = lambda: True
    _fresh_estimate(tracker)
    tracker.step()
    assert tracker._gravity_on
    assert not tracker._startup_done


def test_gravity_is_not_released_without_a_fresh_estimate(tracker_module, monkeypatch):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: True
    tracker.step()  # no PX4 estimate was ever ingested
    tracker.step()
    assert not tracker._gravity_on


def test_homing_stream_is_not_restamped_without_a_fresh_estimate(tracker_module, monkeypatch):
    # `step()` used to call `set_command` every tick with the current wall time, including
    # the hold-last-command branch (no fresh image, no PX4 estimate): the stream's expiry
    # then never fired, so a dead camera or dead telemetry never stopped it.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._startup_done = True
    tracker._gravity_on = True
    tracker.step()
    _, stamp_first = tracker._setpoints.peek()
    tracker.step()
    _, stamp_second = tracker._setpoints.peek()
    assert stamp_second == stamp_first


def test_watchdog_staleness_marks_the_setpoint_stream_fatal(tracker_module, monkeypatch):
    # Spec: the image/telemetry/heartbeat watchdogs were touched but `any_stale` was never
    # consumed anywhere; a dead comms link must stop the normal command.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker._startup_done = True  # the fatal latch is for homing; startup waits within its bounds
    tracker._gravity_on = True
    fresh_now = time.monotonic()
    for name in ("heartbeat", "telemetry", "image"):
        tracker._watchdogs.touch(name, fresh_now)
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: fresh_now)
    tracker.step()
    assert tracker._setpoints.stats.failed is False
    # Time advances well past every watchdog's timeout with no further touches: stale.
    # PX4's heartbeat link is lost, so the run ends as an OFFBOARD_FAILURE, not silently.
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: fresh_now + 10.0)
    with pytest.raises(tracker_module.OffboardFailure, match="watchdog_lost:heartbeat"):
        tracker.step()
    assert tracker._setpoints.stats.failed is True


def test_step_resets_guidance_derivative_history_on_a_clock_discontinuity(tracker_module, monkeypatch):
    from rotorpy.guidance.homing_config import guidance_from_config

    config = load_homing_config(ROOT / "examples" / "camera_homing.jsonc")
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker.guidance = guidance_from_config(config)
    reset_calls = []
    original_reset = tracker.guidance.reset_derivative_history

    def spy_reset():
        reset_calls.append(1)
        original_reset()

    tracker.guidance.reset_derivative_history = spy_reset
    now = time.monotonic()
    for name in ("heartbeat", "telemetry", "image"):
        tracker._watchdogs.touch(name, now)
    tracker._vehicle_state.ingest_attitude(sim_s=10.0, wall_s=10.0, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=10.0, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    # A backward jump to near zero: the simulation restarted (a clock discontinuity).
    tracker._vehicle_state.ingest_attitude(sim_s=0.0, wall_s=10.01, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker.step()
    assert reset_calls == [1]


def test_abort_marks_the_setpoint_stream_fatal_and_reports_target_lost(tracker_module, monkeypatch):
    # Spec (controller decision): the explicit ABORT failure response maps to the T3
    # `SetpointStream` failure-response path and the T1 `TARGET_LOST` outcome, instead
    # of `step()` keeping on issuing (or scoring) a normal command.
    from rotorpy.guidance.tracking import TrackingState

    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker._startup_done = True  # this test is about homing-time ABORT, not startup
    now = time.monotonic()
    for name in ("heartbeat", "telemetry", "image"):
        tracker._watchdogs.touch(name, now)
    tracker._vehicle_state.ingest_attitude(sim_s=10.0, wall_s=now, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=now, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    finish_calls = []
    tracker.telemetry = SimpleNamespace(
        step=lambda **kwargs: None,
        finish=lambda outcome, **kwargs: finish_calls.append((outcome, kwargs)),
    )
    tracker.recorder = None
    aborted_command = SimpleNamespace(
        locked=True, tracking=False, source=None,
        velocity_ned=[0.0, 0.0, 0.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.0,
        tracking_state=TrackingState.ABORT, abort_reason="max_prediction_gap_exceeded",
    )
    tracker.guidance = SimpleNamespace(update=lambda *args, **kwargs: aborted_command)
    assert tracker._setpoints.stats.failed is False
    status = tracker.step()
    assert status == "target_lost"
    assert tracker._setpoints.stats.failed is True
    assert tracker._setpoints.stats.fatal_reason == "target_lost:max_prediction_gap_exceeded"
    assert len(finish_calls) == 1
    outcome, kwargs = finish_calls[0]
    assert outcome.name == "TARGET_LOST"
    assert kwargs["reason"] == "target_lost"
    # One-way, like `SetpointStream.mark_fatal` itself: a later non-ABORT command must
    # not un-fail the stream or re-report a fresh outcome.
    tracker.guidance = SimpleNamespace(update=lambda *args, **kwargs: SimpleNamespace(
        locked=True, tracking=True, source="forward",
        velocity_ned=[1.0, 0.0, 0.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.0,
    ))
    tracker._vehicle_state.ingest_attitude(sim_s=10.1, wall_s=time.monotonic(), quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker.step()
    assert tracker._setpoints.stats.failed is True


def test_guidance_abort_during_startup_does_not_end_the_run(tracker_module, monkeypatch):
    # Finding 3 (fix round 1, promoted): during a long takeoff climb the target can be
    # out of view long enough for guidance's own one-way ABORT latch to fire before the
    # startup gate ever opens. That must not end the run (or even be seen): guidance is
    # not fed anything -- so its internal loss clocks never advance -- until homing has
    # actually started.
    from rotorpy.guidance.tracking import TrackingState

    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker._startup_done = False  # the gate has not opened yet
    now = time.monotonic()
    for name in ("heartbeat", "telemetry", "image"):
        tracker._watchdogs.touch(name, now)
    tracker._vehicle_state.ingest_attitude(sim_s=10.0, wall_s=10.0, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=10.0, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    finish_calls = []
    tracker.telemetry = SimpleNamespace(
        step=lambda **kwargs: None,
        finish=lambda outcome, **kwargs: finish_calls.append((outcome, kwargs)),
    )
    tracker.recorder = None
    guidance_calls = []
    aborted_command = SimpleNamespace(
        locked=True, tracking=False, source=None,
        velocity_ned=[0.0, 0.0, 0.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.0,
        tracking_state=TrackingState.ABORT, abort_reason="max_prediction_gap_exceeded",
    )

    def update(*args, **kwargs):
        guidance_calls.append((args, kwargs))
        return aborted_command

    tracker.guidance = SimpleNamespace(update=update)
    status = tracker.step()
    assert status != "target_lost"
    assert guidance_calls == []  # never fed: the state machine's loss clocks never ran
    assert tracker._setpoints.stats.failed is False
    assert finish_calls == []
    # The gate opening later (a normal, non-abort guidance command) still works: the
    # earlier ABORT left no sticky state behind.
    tracker._motors_ready = lambda: True
    tracker._startup_done = True
    tracker.guidance = SimpleNamespace(update=lambda *args, **kwargs: SimpleNamespace(
        locked=True, tracking=True, source="forward",
        velocity_ned=[1.0, 0.0, 0.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.0,
    ))
    tracker._vehicle_state.ingest_attitude(sim_s=10.1, wall_s=10.1, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=10.1, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    status = tracker.step()
    assert status != "target_lost"
    assert tracker._setpoints.stats.failed is False


def _end_by(tracker, reason):
    """Make the truth side end the flight with `reason` at the next step."""
    if reason == "ground":
        tracker._pose_truth = lambda: (np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    else:
        tracker._scorer = _scorer(timeout_s=0.0)


@pytest.mark.parametrize("reason", ["ground", "timeout"])
def test_tracker_step_reports_flight_end(tracker_module, monkeypatch, reason):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._gravity_on = True
    tracker._startup_done = True
    _end_by(tracker, reason)
    assert tracker.step() == reason


@pytest.mark.parametrize("reason", ["ground", "timeout"])
def test_termination_does_not_need_a_recorder_or_telemetry_or_a_camera(tracker_module, monkeypatch, capsys, reason):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._gravity_on = True
    tracker._startup_done = True
    tracker.recorder = None
    tracker.telemetry = None
    _end_by(tracker, reason)
    assert tracker._observations.take_fresh("forward")[1] is False  # no camera frame ever arrived
    assert tracker.step() == reason
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("RUN_OUTCOME")]
    assert len(lines) == 1
    assert json.loads(lines[0].split(" ", 1)[1])["outcome"] == {"ground": "GROUND_MISS", "timeout": "TIMEOUT"}[reason]


def test_ground_contact_is_never_scored_target_reached_but_a_close_miss_is(tracker_module, monkeypatch):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._gravity_on = True
    tracker._startup_done = True
    tracker._pose_truth = lambda: (np.array([0.0, 0.0, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    assert tracker.step() == "ground"
    assert tracker._lifecycle.outcome.name == "GROUND_MISS"

    near = _new_tracker(tracker_module, monkeypatch)
    near._gravity_on = True
    near._startup_done = True
    # ENU (east 52, north 100, up 0.1): 2 m east of the ground point, 1.25 m off the cylinder surface.
    near._pose_truth = lambda: (np.array([52.0, 100.0, 0.1]), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    assert near.step() == "ground"
    assert near._lifecycle.outcome.name == "TARGET_REACHED"


def test_target_contact_ends_the_run_before_the_ground_and_a_moving_target_is_followed(tracker_module, monkeypatch):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._gravity_on = True
    tracker._startup_done = True
    clock = [10.0]
    monkeypatch.setattr(tracker_module.rospy, "get_time", lambda: clock[0], raising=False)
    # The target drives east at 5 m/s; truth samples come at matching sim times.
    tracker._target_truth = lambda: np.array([100.0, 50.0 + 5.0 * (clock[0] - 10.0), 0.0])
    positions = iter([
        np.array([40.0, 100.0, 1.5]),    # ENU east 40 / north 100 / up 1.5: 10 m west of the target
        np.array([70.0, 100.0, 1.5]),    # 10 m east of the target, which has moved to east 60
    ])
    tracker._pose_truth = lambda: (next(positions), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    assert tracker.step() is None
    clock[0] = 12.0
    assert tracker.step() == "target"
    assert tracker._lifecycle.outcome.name == "TARGET_REACHED"


def test_a_failed_truth_read_is_recoverable_and_bounded(tracker_module, monkeypatch):
    from rotorpy.guidance.run_failure import RecoverableBudget, ServiceUnavailable

    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._gravity_on = True
    tracker._startup_done = True
    tracker._budget = RecoverableBudget(max_consecutive=3, max_total=100)

    def broken():
        raise ServiceUnavailable("get_model_state: no such model")

    tracker._pose_truth = broken
    for _ in range(3):
        assert tracker.step() is None  # control keeps going, the failure is counted
    assert tracker._budget.consecutive == 3
    with pytest.raises(tracker_module.SimulatorFailure, match="recoverable_errors"):
        tracker.step()


@pytest.mark.parametrize("reason", ["ground", "timeout"])
@pytest.mark.parametrize("has_recording", [True, False])
def test_main_stops_after_saving_flight(tracker_module, monkeypatch, capsys, reason, has_recording):
    events = []
    statuses = iter([None, reason])
    fake_tracker = SimpleNamespace(
        step=lambda: events.append("step") or next(statuses),
        write_page=lambda: events.append("write") or ("flight.html" if has_recording else None),
        shutdown=lambda: events.append("shutdown"),
    )
    rospy = tracker_module.rospy
    monkeypatch.setattr(sys, "argv", ["tracker", "--config", "config.jsonc"])
    monkeypatch.setattr(tracker_module, "load_homing_config", lambda path: {"camera_rate_hz": 20})
    monkeypatch.setattr(tracker_module, "gazebo_page_path", lambda path: "flight.html")
    monkeypatch.setattr(tracker_module, "CameraHomingTracker", lambda *args, **kwargs: fake_tracker)
    monkeypatch.setattr(tracker_module.atexit, "register", lambda callback: None)
    monkeypatch.setattr(rospy, "init_node", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(rospy, "on_shutdown", lambda callback: None, raising=False)
    monkeypatch.setattr(rospy, "Rate", lambda rate: SimpleNamespace(sleep=lambda: events.append("sleep")), raising=False)
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False, raising=False)
    tracker_module.main()
    assert events == ["step", "sleep", "step", "write", "shutdown"]
    output = capsys.readouterr()
    assert f"flight finished: {reason}" in output.out
    assert ("no recording was written" in output.err) is not has_recording


def _run_main(tracker_module, monkeypatch, fake_tracker, argv=("tracker", "--config", "config.jsonc")):
    rospy = tracker_module.rospy
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.setattr(tracker_module, "load_homing_config", lambda path: {"camera_rate_hz": 20})
    monkeypatch.setattr(tracker_module, "gazebo_page_path", lambda path: "flight.html")
    monkeypatch.setattr(tracker_module, "CameraHomingTracker", lambda *args, **kwargs: fake_tracker)
    monkeypatch.setattr(tracker_module.atexit, "register", lambda callback: None)
    monkeypatch.setattr(rospy, "init_node", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(rospy, "on_shutdown", lambda callback: None, raising=False)
    monkeypatch.setattr(rospy, "Rate", lambda rate: SimpleNamespace(sleep=lambda: None), raising=False)
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False, raising=False)
    return tracker_module.main()


def test_main_ends_the_run_on_a_fatal_step_error_and_still_flushes(tracker_module, monkeypatch):
    # A fatal `step()` failure ends the run: the tracker handler latches the stream and
    # writes the outcome; main stops (no catch-and-continue), then saves and shuts down.
    events = []

    def step():
        events.append("step")
        raise RuntimeError("boom")

    fake_tracker = SimpleNamespace(
        step=step,
        handle_step_error=lambda exc: events.append(("handle", str(exc))) or False,
        write_page=lambda: events.append("write") or "flight.html",
        shutdown=lambda: events.append("shutdown"),
        exit_code=lambda: 1,
    )
    assert _run_main(tracker_module, monkeypatch, fake_tracker) == 1
    assert events == ["step", ("handle", "boom"), "write", "shutdown"]


def test_main_continues_after_a_recoverable_step_error(tracker_module, monkeypatch):
    events = []
    steps = iter([RuntimeError("blip"), "timeout"])

    def step():
        outcome = next(steps)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    fake_tracker = SimpleNamespace(
        step=step,
        handle_step_error=lambda exc: events.append("recoverable") or True,
        write_page=lambda: events.append("write") or "flight.html",
        shutdown=lambda: events.append("shutdown"),
        exit_code=lambda: 0,
    )
    assert _run_main(tracker_module, monkeypatch, fake_tracker) == 0
    assert events == ["recoverable", "write", "shutdown"]


def test_main_shuts_down_even_if_the_loop_itself_raises(tracker_module, monkeypatch):
    events = []

    def step():
        raise KeyboardInterrupt()

    fake_tracker = SimpleNamespace(
        step=step,
        write_page=lambda: events.append("write"),
        shutdown=lambda: events.append("shutdown"),
    )
    with pytest.raises(KeyboardInterrupt):
        _run_main(tracker_module, monkeypatch, fake_tracker)
    assert events == ["write", "shutdown"]


def test_main_passes_an_explicit_startup_mode_flag(tracker_module, monkeypatch):
    seen = {}
    fake_tracker = SimpleNamespace(
        step=lambda: "timeout", write_page=lambda: "x", shutdown=lambda: None, exit_code=lambda: 0,
    )

    def build(*args, **kwargs):
        seen.update(kwargs)
        return fake_tracker

    rospy = tracker_module.rospy
    monkeypatch.setattr(sys, "argv", ["tracker", "--config", "c.jsonc", "--startup-mode", "takeoff"])
    monkeypatch.setattr(tracker_module, "load_homing_config", lambda path: {"camera_rate_hz": 20})
    monkeypatch.setattr(tracker_module, "gazebo_page_path", lambda path: "flight.html")
    monkeypatch.setattr(tracker_module, "CameraHomingTracker", build)
    monkeypatch.setattr(tracker_module.atexit, "register", lambda callback: None)
    monkeypatch.setattr(rospy, "init_node", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(rospy, "on_shutdown", lambda callback: None, raising=False)
    monkeypatch.setattr(rospy, "Rate", lambda rate: SimpleNamespace(sleep=lambda: None), raising=False)
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False, raising=False)
    tracker_module.main()
    assert seen["startup_mode"] == "takeoff"
    monkeypatch.setattr(sys, "argv", ["tracker", "--config", "c.jsonc", "--startup-mode", "teleport"])
    with pytest.raises(SystemExit):
        tracker_module.main()


def _patch_main_io(tracker_module, monkeypatch, argv):
    rospy = tracker_module.rospy
    monkeypatch.setattr(sys, "argv", list(argv))
    monkeypatch.setattr(tracker_module.atexit, "register", lambda callback: None)
    monkeypatch.setattr(rospy, "init_node", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(rospy, "on_shutdown", lambda callback: None, raising=False)
    monkeypatch.setattr(rospy, "Rate", lambda rate: SimpleNamespace(sleep=lambda: None), raising=False)
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False, raising=False)


def test_main_announces_and_records_an_outcome_when_config_loading_fails(tracker_module, monkeypatch, capsys, tmp_path):
    # Finding 1 (fix round 1): a config load failure must not escape as a bare
    # traceback with no RUN_OUTCOME -- "every run has an outcome" must hold here too.
    page_path = tmp_path / "Results" / "camera_homing_gazebo.html"
    _patch_main_io(tracker_module, monkeypatch, ["tracker", "--config", "bad.jsonc"])
    monkeypatch.setattr(
        tracker_module, "load_homing_config",
        lambda path: (_ for _ in ()).throw(ValueError("missing key: timeout_s")),
    )
    monkeypatch.setattr(tracker_module, "gazebo_page_path", lambda path: page_path)
    exit_code = tracker_module.main()
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "RUN_OUTCOME" in output
    assert "STARTUP_FAILURE" in output
    assert "missing key: timeout_s" in output
    outcome_file = page_path.with_suffix(".outcome.json")
    assert outcome_file.is_file()
    record = json.loads(outcome_file.read_text())
    assert record["outcome"] == "STARTUP_FAILURE"
    assert "missing key: timeout_s" in record["reason"]


def test_main_announces_and_records_an_outcome_when_the_constructor_fails(tracker_module, monkeypatch, capsys, tmp_path):
    # Covers construction failures *after* `load_homing_config` (TruthScorer, telemetry,
    # the recorder, ...) that are not guarded by the tracker's own internal `_connect`
    # try/except -- the gap the reviewer found: these ran unguarded in `main()`.
    page_path = tmp_path / "Results" / "camera_homing_gazebo.html"
    _patch_main_io(tracker_module, monkeypatch, ["tracker", "--config", "config.jsonc"])
    monkeypatch.setattr(tracker_module, "load_homing_config", lambda path: {"camera_rate_hz": 20})
    monkeypatch.setattr(tracker_module, "gazebo_page_path", lambda path: page_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(tracker_module, "CameraHomingTracker", _boom)
    exit_code = tracker_module.main()
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "RUN_OUTCOME" in output
    assert "STARTUP_FAILURE" in output
    assert "disk full" in output
    record = json.loads(page_path.with_suffix(".outcome.json").read_text())
    assert record["outcome"] == "STARTUP_FAILURE"


def test_main_init_failure_outcome_write_never_raises_even_with_an_unwritable_page_path(
    tracker_module, monkeypatch, capsys
):
    # Best-effort: a page path that cannot be written to (for example its parent cannot
    # be created) must not turn the outcome announce itself into an unhandled traceback.
    _patch_main_io(tracker_module, monkeypatch, ["tracker", "--config", "config.jsonc"])
    monkeypatch.setattr(
        tracker_module, "load_homing_config",
        lambda path: (_ for _ in ()).throw(ValueError("bad config")),
    )
    monkeypatch.setattr(tracker_module, "gazebo_page_path", lambda path: "/nonexistent/\0bad/path.html")
    exit_code = tracker_module.main()
    assert exit_code == 1
    assert "RUN_OUTCOME" in capsys.readouterr().out


def test_step_fuses_the_other_cameras_still_recent_cached_image(tracker_module, monkeypatch):
    # Spec: a fresh frame from one camera must not make the *other* camera's stale cached
    # frame appear fresh, but that other camera's still-valid recent observation (within
    # the fusion max age) should still reach guidance instead of being unconditionally
    # dropped to `None`.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker._startup_done = True  # about camera fusion into guidance, not the startup gate
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: 1.0)
    tracker._watchdogs.touch("telemetry", 1.0)
    tracker._vehicle_state.ingest_attitude(sim_s=1.0, wall_s=1.0, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=1.0, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    calls = []

    class FakeGuidance(object):
        def update(self, t, height_m, attitude, velocity_ned, forward_image, upward_image, control_t=None):
            calls.append((forward_image is not None, upward_image is not None))
            return SimpleNamespace(
                locked=False, tracking=False, source=None,
                velocity_ned=[0.0, 0.0, 0.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.0,
            )

    tracker.guidance = FakeGuidance()
    raw = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
    tracker._observations.offer(
        "forward", capture_s=1.0, receipt_s=1.0, raw_bytes=raw, width=2, height=2, encoding="rgb8", step=2 * 3,
    )
    tracker.step()
    assert calls[-1] == (True, False)  # only the forward camera has ever offered anything

    # Shortly after (well within the fusion max age): upward offers a fresh frame, and
    # forward offers nothing new this tick, but its tick-1 frame is still recent.
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: 1.1)
    tracker._watchdogs.touch("telemetry", 1.1)
    tracker._observations.offer(
        "upward", capture_s=1.1, receipt_s=1.1, raw_bytes=raw, width=2, height=2, encoding="rgb8", step=2 * 3,
    )
    tracker.step()
    assert calls[-1] == (True, True)

    # Much later: forward's tick-1 frame has aged out of the fusion window.
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: 10.0)
    tracker._watchdogs.touch("telemetry", 10.0)
    # Keep the estimate itself fresh too (minor c, fix round 1: `have_estimate` now also
    # checks each `VehicleSample` age, not just the combined watchdog).
    tracker._vehicle_state.ingest_attitude(sim_s=10.0, wall_s=10.0, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=10.0, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    tracker._observations.offer(
        "upward", capture_s=10.0, receipt_s=10.0, raw_bytes=raw, width=2, height=2, encoding="rgb8", step=2 * 3,
    )
    tracker.step()
    assert calls[-1] == (False, True)


def _stepping_tracker(tracker_module, monkeypatch, reason, telemetry):
    """A tracker with a PX4 estimate already ingested, matching the old truth pose
    (spawn at (0, 0, 30 m) ENU, identity attitude, zero velocity), so world-frame
    telemetry fields come out the same as when Gazebo truth fed the control loop."""
    tracker = _new_tracker(tracker_module, monkeypatch)
    wall_now = time.monotonic()
    tracker._vehicle_state.ingest_attitude(sim_s=9.9, wall_s=wall_now, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=wall_now, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    tracker._watchdogs.touch("telemetry", wall_now)
    tracker._command = SimpleNamespace(
        locked=True, tracking=True, source="forward", velocity_ned=[1.0, 0.0, 0.0]
    )
    tracker._gravity_on = True
    tracker._startup_done = True
    # The recorder only records; the truth scorer decides when the flight ends.
    tracker.recorder = SimpleNamespace(observe=lambda *args, **kwargs: None)
    tracker.telemetry = telemetry
    _end_by(tracker, reason)
    return tracker


def test_step_records_telemetry_and_a_scored_outcome(tracker_module, monkeypatch):
    calls = []
    telemetry = SimpleNamespace(
        step=lambda **kwargs: calls.append(("step", kwargs)) or True,
        finish=lambda outcome, **kwargs: calls.append(("finish", outcome, kwargs)) or True,
    )
    tracker = _stepping_tracker(tracker_module, monkeypatch, "ground", telemetry)
    assert tracker.step() == "ground"
    assert [call[0] for call in calls] == ["step", "finish"]
    assert calls[0][1]["updated"] is False
    assert calls[1][1].name == "GROUND_MISS"
    assert calls[1][2]["miss_m"] > 50.0


def test_recording_errors_never_reach_the_control_loop(tracker_module, monkeypatch):
    def broken(*args, **kwargs):
        raise OSError("disk full")

    telemetry = SimpleNamespace(step=broken, finish=broken, event=broken)
    tracker = _stepping_tracker(tracker_module, monkeypatch, "timeout", telemetry)
    assert tracker.step() == "timeout"
    tracker.note_failure(RuntimeError("boom"))


def test_tracker_telemetry_round_trips_through_a_real_recorder(tracker_module, monkeypatch, tmp_path):
    from rotorpy.guidance.homing_config import guidance_from_config
    from rotorpy.guidance.outcome import RunOutcome
    from rotorpy.guidance.replay import replay_run
    from rotorpy.guidance.telemetry import read_telemetry, start_run_telemetry

    config = load_homing_config(ROOT / "examples" / "camera_homing.jsonc")
    config.update(image_width_px=64, image_height_px=64)
    recorder = start_run_telemetry(tmp_path, config, environ={"HOMING_SAVE_ALL_FRAMES": "1"})
    tracker = _stepping_tracker(tracker_module, monkeypatch, "timeout", recorder)
    tracker.guidance = guidance_from_config(config)
    tracker._startup_done = True
    raw = np.zeros((64, 64, 3), dtype=np.uint8)
    raw[20:30, 20:30] = (230, 20, 20)
    tracker._command = None
    for camera, receipt_s in (("forward", 9.95), ("upward", 9.96)):
        tracker._observations.offer(
            camera, capture_s=9.9, receipt_s=receipt_s, raw_bytes=raw.tobytes(),
            width=64, height=64, encoding="rgb8", step=64 * 3,
        )
    assert tracker.step() == "timeout"
    recorder.log.close()

    run = read_telemetry(recorder.log.path)
    assert run.metadata["rendering_mode"] == "gpu"
    assert run.ticks[0].image_receipt_s == {"forward": 9.95, "upward": 9.96}
    assert run.ticks[0].position_ned == [0.0, 0.0, -30.0]
    assert run.ticks[0].state == "HOMING"
    assert run.ticks[0].setpoint_stats["publish_count"] == 0  # the 50 Hz timer, not step(), publishes
    assert run.outcome["outcome"] == RunOutcome.TIMEOUT.value
    assert run.outcome["reason"] == "timeout"
    assert replay_run(recorder.log.path).matches


def test_startup_heading_comes_from_the_vehicle_not_the_target(tracker_module, monkeypatch):
    from scipy.spatial.transform import Rotation

    tracker = _stepping_tracker(tracker_module, monkeypatch, "timeout", None)
    heading = np.deg2rad(200.0)
    tracker._startup_yaw = None
    tracker._vehicle_state.ingest_attitude(
        sim_s=9.9, wall_s=9.9, quat_estimator_xyzw=Rotation.from_euler("z", heading).as_quat()
    )
    tracker.step()
    assert np.isclose(np.cos(tracker._startup_yaw - heading), 1.0)


# --- fix round 2: discontinuity ownership, image dropout not fatal -----------------------


class _FakeGuidance(object):
    def __init__(self):
        self.resets = 0

    def reset_derivative_history(self):
        self.resets += 1

    def update(self, t, height_m, attitude, velocity_ned, forward_image, upward_image, control_t=None):
        return SimpleNamespace(
            locked=False, tracking=False, source=None,
            velocity_ned=[1.0, 0.0, 0.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.0,
        )


def _ingest_estimate(tracker, wall_s):
    tracker._vehicle_state.ingest_attitude(sim_s=wall_s, wall_s=wall_s, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=wall_s, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )


def _offer_frame(tracker, receipt_s):
    raw = np.zeros((2, 2, 3), dtype=np.uint8).tobytes()
    tracker._observations.offer(
        "forward", capture_s=receipt_s, receipt_s=receipt_s, raw_bytes=raw, width=2, height=2,
        encoding="rgb8", step=2 * 3,
    )
    tracker._watchdogs.touch("image", receipt_s)


def test_setpoint_publisher_cannot_swallow_the_discontinuity_before_step_sees_it(tracker_module, monkeypatch):
    # The 50 Hz publisher thread calls `_publish_setpoint` -> `sample_at(None, ...)`; before
    # this fix that call consumed the sticky discontinuity, so `step()` never reset
    # guidance's derivative history.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker.guidance = _FakeGuidance()
    tracker._send = lambda command: None
    now = time.monotonic()
    for name in ("heartbeat", "telemetry", "image"):
        tracker._watchdogs.touch(name, now)
    _ingest_estimate(tracker, 10.0)
    tracker._vehicle_state.ingest_attitude(sim_s=0.0, wall_s=10.01, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._publish_setpoint(_FakeGuidance().update(0, 0, 0, 0, 0, 0))
    tracker.step()
    assert tracker.guidance.resets == 1
    tracker.step()
    assert tracker.guidance.resets == 1


def _dropout_tracker(tracker_module, monkeypatch, published, clock):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker._startup_done = True  # these tests are about homing-time link loss
    tracker._gravity_on = True
    tracker.guidance = _FakeGuidance()
    tracker._setpoints = tracker_module.SetpointStream(
        publish=published.append, expiry_s=0.5, now=lambda: clock[0]
    )
    return tracker


def _tick_at(tracker_module, monkeypatch, tracker, clock, t, link_touches):
    clock[0] = t
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: t)
    for name in link_touches:
        tracker._watchdogs.touch(name, t)


def test_camera_dropout_is_not_fatal_and_normal_commands_resume_on_recovery(tracker_module, monkeypatch):
    published, clock = [], [0.0]
    tracker = _dropout_tracker(tracker_module, monkeypatch, published, clock)
    _tick_at(tracker_module, monkeypatch, tracker, clock, 1.0, ("heartbeat", "telemetry"))
    _ingest_estimate(tracker, 1.0)
    _offer_frame(tracker, 1.0)
    tracker.step()
    assert tracker._setpoints.tick(now_s=1.1) is not None

    # The camera goes quiet well past the image watchdog timeout; telemetry/heartbeat live.
    _tick_at(tracker_module, monkeypatch, tracker, clock, 3.0, ("heartbeat", "telemetry"))
    _ingest_estimate(tracker, 3.0)
    tracker.step()
    assert tracker._setpoints.stats.failed is False
    assert tracker._setpoints.stats.fatal_reason is None
    assert tracker._setpoints.tick(now_s=3.0) is None  # expired, not fatal
    assert tracker._stale_links == ["image"]  # still visible

    # A fresh frame arrives: the stream resumes normal commands.
    _tick_at(tracker_module, monkeypatch, tracker, clock, 3.2, ("heartbeat", "telemetry"))
    _ingest_estimate(tracker, 3.2)
    _offer_frame(tracker, 3.2)
    tracker.step()
    assert tracker._setpoints.stats.failed is False
    assert tracker._setpoints.tick(now_s=3.25) is not None
    assert tracker._stale_links == []


def test_telemetry_loss_is_fatal_and_records_why(tracker_module, monkeypatch):
    published, clock = [], [0.0]
    tracker = _dropout_tracker(tracker_module, monkeypatch, published, clock)
    _tick_at(tracker_module, monkeypatch, tracker, clock, 1.0, ("heartbeat", "telemetry", "image"))
    _ingest_estimate(tracker, 1.0)
    _offer_frame(tracker, 1.0)
    tracker.step()
    # Telemetry goes silent while heartbeat and image stay alive.
    _tick_at(tracker_module, monkeypatch, tracker, clock, 2.0, ("heartbeat", "image"))
    with pytest.raises(tracker_module.EstimatorFailure, match="watchdog_lost:telemetry"):
        tracker.step()
    assert tracker._setpoints.stats.failed is True
    assert tracker._setpoints.stats.fatal_reason == "watchdog_lost:telemetry"
    # Latched: even a fresh frame and recovered telemetry do not bring normal commands back.
    _tick_at(tracker_module, monkeypatch, tracker, clock, 2.1, ("heartbeat", "telemetry"))
    _ingest_estimate(tracker, 2.1)
    _offer_frame(tracker, 2.1)
    tracker.step()
    assert tracker._setpoints.tick(now_s=2.15) is None
    assert tracker._setpoints.stats_snapshot()["fatal_reason"] == "watchdog_lost:telemetry"


def test_never_seen_link_does_not_latch_fatal_at_startup(tracker_module, monkeypatch):
    # Before the first PX4 attitude/local-position message arrives telemetry is "stale"
    # (never touched); that is startup, not a lost link, and must not latch forever.
    published, clock = [], [0.0]
    tracker = _dropout_tracker(tracker_module, monkeypatch, published, clock)
    _tick_at(tracker_module, monkeypatch, tracker, clock, 1.0, ("heartbeat",))
    tracker.step()
    assert tracker._setpoints.stats.failed is False
    assert tracker._setpoints.tick(now_s=1.6) is None  # past expiry: stream goes quiet


def test_recording_carries_the_stale_links_and_fatal_reason_in_setpoint_stats(tracker_module, monkeypatch):
    published, clock = [], [0.0]
    tracker = _dropout_tracker(tracker_module, monkeypatch, published, clock)
    seen = []
    tracker.telemetry = SimpleNamespace(step=lambda **kwargs: seen.append(kwargs))
    tracker.recorder = None
    _tick_at(tracker_module, monkeypatch, tracker, clock, 1.0, ("heartbeat", "telemetry"))
    _ingest_estimate(tracker, 1.0)
    tracker.step()
    assert seen[-1]["setpoint_stats"]["stale_links"] == ["image"]
    assert seen[-1]["setpoint_stats"]["fatal_reason"] is None


# --- plan phase 8: startup gate, explicit stream policy, outcomes, shutdown --------------


def test_startup_stream_runs_on_the_estimate_alone_before_the_first_image(tracker_module, monkeypatch):
    # Carry-over: the placeholder command used to be stamped once and then expire, so the
    # stream went silent whenever the PX4 estimate preceded the first camera image.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    clock = [100.0]
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: clock[0])
    _fresh_estimate(tracker, 100.0)
    stamps = []
    for step in range(30):  # 1.5 s of steps at 20 Hz: three expiry intervals of 0.5 s
        clock[0] = 100.0 + 0.05 * step
        _fresh_estimate(tracker, clock[0])
        tracker.step()
        command, stamp = tracker._setpoints.peek()
        stamps.append(stamp)
        assert isinstance(command, tracker_module.StartupCommand)
        assert stamp == pytest.approx(clock[0])  # restamped every step: never expires on its own
    assert command.velocity_ned == (0.0, 0.0, -3.0)  # airborne spawn, gravity off: the legacy climb hold
    assert command.yaw_rad == 0.0  # the vehicle's own heading, never a target direction


def test_startup_stream_goes_silent_when_the_estimate_stops_being_fresh(tracker_module, monkeypatch):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker._setpoints = tracker_module.SetpointStream(publish=lambda command: None, expiry_s=0.5)
    clock = [100.0]
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: clock[0])
    _fresh_estimate(tracker, 100.0)
    tracker.step()
    _, stamp = tracker._setpoints.peek()
    clock[0] = 105.0  # telemetry is silent (never re-touched): the estimate is no longer fresh
    tracker.step()
    assert tracker._setpoints.peek()[1] == stamp  # not restamped
    assert tracker._setpoints.tick(now_s=105.0) is None  # expired: nothing is sent


def test_have_estimate_requires_fresh_position_not_just_the_combined_watchdog(tracker_module, monkeypatch):
    # Minor (c), fix round 1: the single "telemetry" watchdog is touched by *either*
    # ATTITUDE_QUATERNION or LOCAL_POSITION_NED messages, so it can stay "fresh" while
    # one of the two has actually gone stale. `have_estimate` must also check each
    # `VehicleSample` age (`attitude_age_s`/`position_age_s`) against
    # `PROPOSED_TELEMETRY_TIMEOUT_S`, not rely on the combined watchdog alone.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker._setpoints = tracker_module.SetpointStream(publish=lambda command: None, expiry_s=0.5)
    clock = [100.0]
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: clock[0])
    _fresh_estimate(tracker, 100.0)
    tracker.step()
    _, stamp = tracker._setpoints.peek()
    # Position is never re-ingested, but attitude keeps arriving and re-touches the
    # single combined "telemetry" watchdog all on its own, past PROPOSED_TELEMETRY_TIMEOUT_S.
    clock[0] = 100.6
    tracker._vehicle_state.ingest_attitude(sim_s=clock[0], wall_s=clock[0], quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._watchdogs.touch("telemetry", clock[0])
    tracker.step()
    assert tracker._setpoints.peek()[1] == stamp  # not restamped: the estimate is not fresh


def test_nothing_is_streamed_during_startup_without_an_estimate(tracker_module, monkeypatch):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker.step()
    assert tracker._setpoints.peek() == (None, None)


def test_the_homing_command_never_transmits_while_the_gate_is_still_open(tracker_module, monkeypatch):
    # Finding 3 (fix round 1): guidance is not even fed while the gate is closed (not
    # merely "computed but not sent" as before), so the held command is the placeholder.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    guidance_calls = []
    tracker.guidance = SimpleNamespace(update=lambda *a, **k: guidance_calls.append(1) or _FakeGuidance().update(
        *a, **k
    ))
    _fresh_estimate(tracker)
    _offer_frame(tracker, time.monotonic())
    tracker.step()
    command, _ = tracker._setpoints.peek()
    assert isinstance(command, tracker_module.StartupCommand)
    assert guidance_calls == []  # guidance never ran while the gate is still closed
    assert np.array_equal(tracker._command.velocity_ned, np.zeros(3))  # the placeholder



def _takeoff_tracker(tracker_module, monkeypatch, clock):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker.startup_mode = tracker_module.StartupMode.TAKEOFF
    tracker.guidance = _FakeGuidance()
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: clock[0])
    return tracker


def test_takeoff_mode_reaches_homing_through_the_gate_without_touching_gravity(tracker_module, monkeypatch):
    clock = [100.0]
    tracker = _takeoff_tracker(tracker_module, monkeypatch, clock)
    gravity_calls = []
    tracker._set_gravity = lambda enabled: gravity_calls.append(enabled)
    tracker._motors_ready = lambda: pytest.fail("takeoff mode must not wait on the gravity-hold motor check")
    engage_calls = []
    tracker._engage = lambda: engage_calls.append(clock[0])
    for step in range(120):
        clock[0] = 100.0 + 0.05 * step
        _fresh_estimate(tracker, clock[0])
        _offer_frame(tracker, clock[0])
        tracker.step()
        if tracker._startup_done:
            break
    assert tracker._startup_done
    assert gravity_calls == []
    assert engage_calls == []  # `_flight_state` already reports armed + offboard
    command, _ = tracker._setpoints.peek()
    assert command.velocity_ned == [1.0, 0.0, 0.0]  # the guidance command takes over


def test_arming_is_requested_through_the_gate_and_bounded(tracker_module, monkeypatch):
    clock = [100.0]
    tracker = _takeoff_tracker(tracker_module, monkeypatch, clock)
    tracker._flight_state = lambda: False
    engage_calls = []
    tracker._engage = lambda: engage_calls.append(clock[0])
    with pytest.raises(tracker_module.StartupFailure, match="offboard_not_confirmed"):
        for step in range(2000):
            clock[0] = 100.0 + 0.05 * step
            _fresh_estimate(tracker, clock[0])
            tracker.step()
    assert 1 <= len(engage_calls) <= 20  # PROPOSED_MAX_ARM_ATTEMPTS, never unbounded


def test_armed_and_offboard_are_fed_as_separate_flags_to_the_startup_gate(tracker_module, monkeypatch):
    # Minor (c), fix round 1: `_hb_armed` (the raw PX4 ARM state) and `_offboard`
    # (armed AND in offboard mode) are separate data the tracker already has -- the gate
    # must see each on its own field, not the same combined flag copied into both.
    from rotorpy.guidance.startup import StartupDecision, StartupPhase

    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker._hb_armed = True
    tracker._flight_state = lambda: False  # armed, but not (yet) in offboard mode
    captured = []

    class FakeGate(object):
        phase = StartupPhase.MOTOR_START

        def update(self, inputs):
            captured.append(inputs)
            return StartupDecision(phase=StartupPhase.MOTOR_START)

    tracker._gate = FakeGate()
    tracker.step()
    assert len(captured) == 1
    assert captured[0].armed is True
    assert captured[0].offboard is False


def test_startup_failure_latches_the_stream_and_writes_exactly_one_outcome(tracker_module, monkeypatch, capsys):
    clock = [100.0]
    tracker = _takeoff_tracker(tracker_module, monkeypatch, clock)
    calls = []
    tracker.telemetry = SimpleNamespace(
        step=lambda **kwargs: True,
        finish=lambda outcome, **kwargs: calls.append((outcome, kwargs)) or True,
        close=lambda **kwargs: calls.append("close"),
    )
    tracker.step()  # the gate starts waiting for an estimate that never arrives
    clock[0] = 100.0 + 70.0  # past the 60 s estimate wait
    with pytest.raises(tracker_module.StartupFailure) as info:
        tracker.step()
    assert tracker.handle_step_error(info.value) is False
    assert tracker.handle_step_error(info.value) is False  # a second report adds nothing
    assert tracker._setpoints.stats.failed
    assert tracker._setpoints.stats.fatal_reason.startswith("startup_failure")
    outcomes = [call for call in calls if call != "close"]
    assert [o[0].name for o in outcomes] == ["STARTUP_FAILURE"]
    assert outcomes[0][1]["reason"] == "estimate_timeout"
    assert tracker.exit_code() == 1


def test_a_failed_init_is_reported_as_startup_failure_by_the_first_step(tracker_module, monkeypatch):
    rospy = tracker_module.rospy
    monkeypatch.setattr(rospy, "Timer", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Duration", lambda seconds: seconds, raising=False)
    monkeypatch.setattr(rospy, "Subscriber", lambda *a, **k: None, raising=False)

    def never_up(name, timeout=None):
        raise RuntimeError("timeout waiting for {}".format(name))

    monkeypatch.setattr(rospy, "wait_for_service", never_up, raising=False)
    config = load_homing_config(ROOT / "examples" / "camera_homing.jsonc")
    tracker = tracker_module.CameraHomingTracker(config, "hummingbird_0")  # does not raise or hang
    with pytest.raises(tracker_module.StartupFailure, match="service_timeout:/gazebo/get_link_properties"):
        tracker.step()
    assert tracker.handle_step_error(tracker_module.StartupFailure("x")) is False
    tracker.shutdown()  # drains the queued outcome
    assert tracker._lifecycle.outcome.name == "STARTUP_FAILURE"


def test_a_missing_heartbeat_is_a_bounded_startup_failure(tracker_module, monkeypatch):
    rospy = tracker_module.rospy
    monkeypatch.setattr(rospy, "wait_for_service", lambda name, timeout=None: None, raising=False)
    monkeypatch.setattr(rospy, "Subscriber", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Timer", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Duration", lambda seconds: seconds, raising=False)
    monkeypatch.setattr(rospy, "ServiceProxy", lambda name, _type: (lambda *args: None), raising=False)
    monkeypatch.setattr(tracker_module.CameraHomingTracker, "_set_gravity", lambda self, enabled: None)
    mav = SimpleNamespace(target_system=1, wait_heartbeat=lambda timeout=None: None)
    monkeypatch.setattr(tracker_module.mavutil, "mavlink_connection", lambda *args, **kwargs: mav)
    config = load_homing_config(ROOT / "examples" / "camera_homing.jsonc")
    tracker = tracker_module.CameraHomingTracker(config, "hummingbird_0")
    with pytest.raises(tracker_module.StartupFailure, match="heartbeat_timeout"):
        tracker.step()


def test_airborne_spawn_is_an_explicit_mode_takeoff_never_disables_gravity(tracker_module, monkeypatch):
    rospy = tracker_module.rospy
    gravity = []
    monkeypatch.setattr(rospy, "wait_for_service", lambda name, timeout=None: None, raising=False)
    monkeypatch.setattr(rospy, "Subscriber", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Timer", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Duration", lambda seconds: seconds, raising=False)
    monkeypatch.setattr(rospy, "ServiceProxy", lambda name, _type: (lambda *args: None), raising=False)
    monkeypatch.setattr(
        tracker_module.CameraHomingTracker, "_set_gravity", lambda self, enabled: gravity.append(enabled)
    )
    mav = SimpleNamespace(target_system=1, wait_heartbeat=lambda timeout=None: object())
    monkeypatch.setattr(tracker_module.mavutil, "mavlink_connection", lambda *args, **kwargs: mav)
    config = load_homing_config(ROOT / "examples" / "camera_homing.jsonc")
    airborne = tracker_module.CameraHomingTracker(config, "hummingbird_0", startup_mode="airborne_spawn")
    assert gravity == [False]
    takeoff = tracker_module.CameraHomingTracker(config, "hummingbird_0", startup_mode="takeoff")
    assert gravity == [False]  # unchanged: takeoff never touches gravity
    assert takeoff.startup_mode is tracker_module.StartupMode.TAKEOFF
    with pytest.raises(ValueError, match="startup mode"):
        tracker_module.CameraHomingTracker(config, "hummingbird_0", startup_mode="teleport")
    airborne.shutdown()
    takeoff.shutdown()


def test_handle_step_error_bounds_recoverable_errors_and_classifies_the_rest(tracker_module, monkeypatch):
    from rotorpy.guidance.run_failure import RecoverableBudget, RecoverableError

    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._budget = RecoverableBudget(max_consecutive=2, max_total=10)
    assert tracker.handle_step_error(RecoverableError("blip")) is True
    assert tracker.handle_step_error(RecoverableError("blip")) is True
    assert not tracker._setpoints.stats.failed
    assert tracker.handle_step_error(RecoverableError("blip")) is False  # escalated
    assert tracker._setpoints.stats.failed
    assert tracker._lifecycle.outcome.name == "SIMULATOR_FAILURE"

    unknown = _new_tracker(tracker_module, monkeypatch)
    assert unknown.handle_step_error(KeyError("boom")) is False
    assert unknown._setpoints.stats.fatal_reason.startswith("step_exception")
    assert unknown._lifecycle.outcome.name == "SIMULATOR_FAILURE"
    assert unknown._lifecycle.outcome_reason.startswith("unexpected:KeyError")

    offboard = _new_tracker(tracker_module, monkeypatch)
    offboard.handle_step_error(tracker_module.OffboardFailure("offboard_lost"))
    assert offboard._lifecycle.outcome.name == "OFFBOARD_FAILURE"


def test_shutdown_flushes_and_closes_everything_even_when_sinks_fail(tracker_module, monkeypatch, capsys):
    tracker = _new_tracker(tracker_module, monkeypatch)
    closed = []

    class FailingTelemetry(object):
        def step(self, **kwargs):
            raise OSError("disk full")

        def finish(self, outcome, **kwargs):
            raise OSError("disk full")

        def close(self, flush_frames=False):
            closed.append(("telemetry", flush_frames))
            raise OSError("close failed")

    tracker.telemetry = FailingTelemetry()
    recording = SimpleNamespace(close=lambda timeout=None: closed.append(("worker", timeout)))
    tracker._recording = recording
    tracker.shutdown()
    tracker.shutdown()  # idempotent
    assert closed == [("worker", 5.0), ("telemetry", True)]
    # No outcome had been decided: shutdown records exactly one infrastructure outcome,
    # printed even though the telemetry sink failed.
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("RUN_OUTCOME")]
    assert len(lines) == 1
    payload = json.loads(lines[0].split(" ", 1)[1])
    assert payload["outcome"] == "SIMULATOR_FAILURE" and payload["reason"] == "shutdown_before_outcome"


def test_shutdown_after_an_orderly_outcome_keeps_it(tracker_module, monkeypatch):
    calls = []
    telemetry = SimpleNamespace(
        step=lambda **kwargs: True,
        finish=lambda outcome, **kwargs: calls.append(outcome.name) or True,
        close=lambda flush_frames=False: calls.append("close"),
    )
    tracker = _stepping_tracker(tracker_module, monkeypatch, "ground", telemetry)
    assert tracker.step() == "ground"
    tracker._recording = SimpleNamespace(close=lambda timeout=None: None)
    tracker.shutdown()
    assert calls == ["GROUND_MISS", "close"]


def test_shutdown_salvages_a_pending_outcome_instead_of_overwriting_it(tracker_module, monkeypatch, capsys):
    # Minor (a), fix round 1: `_recording.close(timeout)` can return before its worker
    # thread has drained a real, already-decided outcome still sitting in its queue (it
    # raced the timeout). Shutdown must not shadow that outcome with a wrong
    # SIMULATOR_FAILURE "shutdown_before_outcome" -- it must write the queued one.
    tracker = _new_tracker(tracker_module, monkeypatch)
    finish_calls = []
    tracker.telemetry = SimpleNamespace(
        step=lambda **kwargs: None,
        finish=lambda outcome, **kwargs: finish_calls.append((outcome, kwargs)) or True,
    )
    tracker.recorder = None
    queued = []
    # A worker that never drains anything: `submit` only queues, `close` times out
    # without processing the queue (exactly the race this fix targets).
    tracker._recording = SimpleNamespace(submit=queued.append, close=lambda timeout=None: None)
    tracker._submit_outcome("timeout")
    assert len(queued) == 1  # queued, but never processed by any worker thread
    tracker.shutdown()
    assert len(finish_calls) == 1
    outcome, kwargs = finish_calls[0]
    assert outcome.name == "TIMEOUT"
    assert kwargs["reason"] == "timeout"
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("RUN_OUTCOME")]
    assert len(lines) == 1
    assert json.loads(lines[0].split(" ", 1)[1])["outcome"] == "TIMEOUT"


def test_shutdown_is_guarded_by_an_atomic_lock(tracker_module, monkeypatch):
    # Minor (a), fix round 1: two concurrent callers (for example `rospy.on_shutdown` and
    # `atexit`) must never both enter the shutdown body -- a bare flag has a
    # check-then-set race window a lock does not.
    import threading

    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker.telemetry = None
    tracker.recorder = None
    entered = threading.Event()
    proceed = threading.Event()
    calls = []

    def slow_close(timeout=None):
        entered.set()
        proceed.wait(timeout=2.0)
        calls.append(1)

    tracker._recording = SimpleNamespace(close=slow_close)
    threads = [threading.Thread(target=tracker.shutdown) for _ in range(2)]
    for thread in threads:
        thread.start()
    assert entered.wait(timeout=2.0)
    time.sleep(0.05)  # give a racing second caller every chance to also enter
    proceed.set()
    for thread in threads:
        thread.join(timeout=2.0)
    assert calls == [1]  # the body ran exactly once


def test_telemetry_ticks_carry_the_phase_transmitted_setpoint_and_failure_reason(tracker_module, monkeypatch):
    seen = []
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker.telemetry = SimpleNamespace(step=lambda **kwargs: seen.append(kwargs) or True)
    tracker._motors_ready = lambda: False
    tracker.guidance = _FakeGuidance()
    tracker._last_transmitted = [0.0, 0.0, -3.0]
    _fresh_estimate(tracker)
    tracker._setpoints.mark_fatal("watchdog_lost:telemetry")
    tracker.step()
    tick = seen[-1]
    assert tick["state"] in {"ENGAGE", "MOTOR_START"}
    assert tick["transmitted_setpoint"] == [0.0, 0.0, -3.0]
    assert tick["failure_reason"] == "watchdog_lost:telemetry"
    assert tick["attitude_age_s"] is not None and tick["position_age_s"] is not None


def test_target_truth_reads_the_target_model_on_the_truth_side_only(tracker_module):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    assert tracker._target_truth() is None  # no model-state service: scorer falls back to the config
    states = {}

    def model_state(name, relative):
        states["asked"] = name
        pose = SimpleNamespace(position=SimpleNamespace(x=50.0, y=100.0, z=1.5))
        return SimpleNamespace(success=True, pose=pose, status_message="")

    tracker._model_state = model_state
    tracker._target_height_m = 3.0
    ground = tracker._target_truth()
    assert states["asked"] == "homing_target"
    # ENU centre (50, 100, 1.5) -> NED (100, 50, -1.5); the ground contact is half the height below.
    assert ground == pytest.approx([100.0, 50.0, 0.0])


def test_launcher_selects_the_startup_mode_explicitly():
    launcher = (ROOT / "Dockerfiles" / "sitl_hummingbird_run.sh").read_text()
    assert 'HOMING_STARTUP_MODE="${HOMING_STARTUP_MODE:-airborne_spawn}"' in launcher
    assert '--startup-mode "${HOMING_STARTUP_MODE}"' in launcher
    assert "takeoff|airborne_spawn" in launcher
    # The airborne spawn (the one scene_consistency validates) comes first; the takeoff
    # spawn on the ground is a separate branch.
    assert launcher.index("-z 30 -Y") < launcher.index('--model-name="hummingbird_0" -x 0 -y 0 -z 0.2 -Y')
    runner = (ROOT / "Dockerfiles" / "run_hummingbird_sitl.sh").read_text()
    assert "HOMING_STARTUP_MODE" in runner
