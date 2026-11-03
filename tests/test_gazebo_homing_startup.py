"""Verify the Gazebo startup hold and PX4 heartbeat handling without ROS."""

import importlib.util
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
    monkeypatch.setattr(rospy, "wait_for_service", lambda name: events.append(("wait", name)), raising=False)
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
    mav = SimpleNamespace(target_system=1, wait_heartbeat=lambda: None)
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


def test_startup_commands_climb_then_hold_before_homing(tracker_module):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._gravity_on = False
    tracker._startup_done = False
    tracker._startup_yaw = 0.3
    tracker._startup_height = 30.0
    sent = []
    tracker._mav = SimpleNamespace(
        target_system=1, target_component=1,
        mav=SimpleNamespace(set_position_target_local_ned_send=lambda *args: sent.append(args)),
    )
    command = SimpleNamespace(
        velocity_ned=[10.0, 5.0, 3.0], acceleration_ned=[0.0, 0.0, 0.0], yaw_rad=0.8
    )
    tracker._send(command, 30.0)
    assert sent[-1][8:11] == (0.0, 0.0, -tracker_module._STARTUP_CLIMB_SPEED)
    assert sent[-1][14] == 0.3
    tracker._gravity_on = True
    tracker._send(command, 32.0)
    assert sent[-1][8:11] == (0.0, 0.0, 2.0)
    tracker._send(command, 29.0)
    assert sent[-1][8:11] == (0.0, 0.0, -1.0)
    tracker._startup_done = True
    tracker._send(command, 30.0)
    assert sent[-1][8:11] == (10.0, 5.0, 3.0)
    assert sent[-1][14] == 0.8


def test_mav_access_is_serialised_across_threads(tracker_module):
    # pymavlink is not thread-safe: the 50 Hz setpoint-publisher thread's `_send` and the
    # step thread's `_poll_mavlink` must never run inside the same connection at once.
    import threading

    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._gravity_on = True
    tracker._startup_done = True
    tracker._startup_yaw = 0.0
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
    send_thread = threading.Thread(target=lambda: tracker._send(command, 30.0))
    poll_thread = threading.Thread(target=tracker._poll_mavlink)
    send_thread.start()
    poll_thread.start()
    send_thread.join()
    poll_thread.join()
    assert overlap == []


def test_startup_requires_a_sustained_stable_hover(tracker_module):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._startup_done = False
    tracker._startup_height = 30.0
    tracker._settled_since = None
    tracker._hold_start_time = 0.0
    tracker._update_startup_hold(1.0, 30.0, 0.0)
    tracker._update_startup_hold(1.8, 31.5, 0.0)
    tracker._update_startup_hold(2.0, 30.0, 0.0)
    tracker._update_startup_hold(2.9, 30.0, 0.0)
    assert not tracker._startup_done
    tracker._update_startup_hold(3.0, 30.0, 0.0)
    assert tracker._startup_done
    tracker._startup_done = False
    with pytest.raises(RuntimeError, match="did not stabilize"):
        tracker._update_startup_hold(21.0, 35.0, 1.0)


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
    tracker._sent = 20
    tracker._flight_state = lambda: armed
    tracker._gravity_on = False
    tracker._motor_start_time = None
    tracker._startup_done = False
    tracker._hold_start_time = None
    tracker._settled_since = None
    tracker._hb_mode = None
    tracker._target_ground_ned = np.array([100.0, 50.0, 0.0])
    tracker.recorder = None
    tracker.telemetry = None
    tracker._pose_truth = lambda: (np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    tracker._set_gravity = lambda enabled: setattr(tracker, "_gravity_on", enabled)
    monkeypatch.setattr(tracker_module.rospy, "get_time", lambda: 10.0, raising=False)
    return tracker


def test_gravity_stays_off_until_armed_offboard_motors_spin(tracker_module, monkeypatch):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    tracker.step()
    assert not tracker._gravity_on
    tracker._motors_ready = lambda: True
    tracker.step()
    assert tracker._gravity_on
    assert not tracker._startup_done


def test_step_does_not_restamp_the_setpoint_stream_without_a_fresh_estimate(tracker_module, monkeypatch):
    # Before this fix, `step()` called `set_command` every tick with the current wall
    # time, including the hold-last-command branch (no fresh image, no PX4 estimate):
    # the setpoint stream's expiry then never fired, so a dead camera or dead telemetry
    # never stopped the stream.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    # No PX4 estimate is ever ingested: `have_estimate` is False on every tick.
    tracker.step()
    _, stamp_first = tracker._setpoints.peek()
    assert stamp_first is not None
    tracker.step()
    _, stamp_second = tracker._setpoints.peek()
    assert stamp_second == stamp_first


def test_watchdog_staleness_marks_the_setpoint_stream_fatal(tracker_module, monkeypatch):
    # Spec: the image/telemetry/heartbeat watchdogs were touched but `any_stale` was never
    # consumed anywhere; a dead comms link must stop the normal command.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    fresh_now = time.monotonic()
    for name in ("heartbeat", "telemetry", "image"):
        tracker._watchdogs.touch(name, fresh_now)
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: fresh_now)
    tracker.step()
    assert tracker._setpoints.stats.failed is False
    # Time advances well past every watchdog's timeout with no further touches: stale.
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: fresh_now + 10.0)
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
    tracker._vehicle_state.ingest_attitude(sim_s=10.1, wall_s=10.1, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker.step()
    assert tracker._setpoints.stats.failed is True


@pytest.mark.parametrize("reason", ["ground", "timeout"])
def test_tracker_step_reports_flight_end(tracker_module, monkeypatch, reason):
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._gravity_on = True
    tracker._startup_done = True
    tracker.recorder = SimpleNamespace(observe=lambda *args: reason)
    assert tracker.step() == reason


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
    monkeypatch.setattr(tracker_module, "CameraHomingTracker", lambda *args: fake_tracker)
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


def test_main_marks_the_setpoint_stream_fatal_on_a_step_exception(tracker_module, monkeypatch):
    # A fatal `step()` failure must stop the normal command just like an expired or
    # watchdog-stale one, not leave the stream streaming whatever was last held forever.
    events = []
    outcomes = iter([RuntimeError("boom"), "timeout"])

    def step():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    fake_tracker = SimpleNamespace(
        step=step,
        write_page=lambda: events.append("write") or "flight.html",
        shutdown=lambda: events.append("shutdown"),
        note_failure=lambda exc: events.append(("note_failure", str(exc))),
        mark_fatal=lambda reason=None: events.extend(["mark_fatal", reason]),
    )
    rospy = tracker_module.rospy
    monkeypatch.setattr(sys, "argv", ["tracker", "--config", "config.jsonc"])
    monkeypatch.setattr(tracker_module, "load_homing_config", lambda path: {"camera_rate_hz": 20})
    monkeypatch.setattr(tracker_module, "gazebo_page_path", lambda path: "flight.html")
    monkeypatch.setattr(tracker_module, "CameraHomingTracker", lambda *args: fake_tracker)
    monkeypatch.setattr(tracker_module.atexit, "register", lambda callback: None)
    monkeypatch.setattr(rospy, "init_node", lambda *args, **kwargs: None, raising=False)
    monkeypatch.setattr(rospy, "on_shutdown", lambda callback: None, raising=False)
    monkeypatch.setattr(rospy, "Rate", lambda rate: SimpleNamespace(sleep=lambda: None), raising=False)
    monkeypatch.setattr(rospy, "is_shutdown", lambda: False, raising=False)
    tracker_module.main()
    assert "mark_fatal" in events
    assert events.index("mark_fatal") < events.index("write")
    assert events[events.index("mark_fatal") + 1] == "step_exception: boom"


def test_step_fuses_the_other_cameras_still_recent_cached_image(tracker_module, monkeypatch):
    # Spec: a fresh frame from one camera must not make the *other* camera's stale cached
    # frame appear fresh, but that other camera's still-valid recent observation (within
    # the fusion max age) should still reach guidance instead of being unconditionally
    # dropped to `None`.
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker._motors_ready = lambda: False
    monkeypatch.setattr(tracker_module.time, "monotonic", lambda: 1.0)
    tracker._watchdogs.touch("telemetry", 1.0)
    tracker._vehicle_state.ingest_attitude(sim_s=1.0, wall_s=1.0, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=1.0, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    calls = []

    class FakeGuidance(object):
        def update(self, t, height_m, attitude, velocity_ned, forward_image, upward_image):
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
    tracker._vehicle_state.ingest_attitude(sim_s=9.9, wall_s=9.9, quat_estimator_xyzw=[0.0, 0.0, 0.0, 1.0])
    tracker._vehicle_state.ingest_local_position(
        wall_s=9.9, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    tracker._watchdogs.touch("telemetry", time.monotonic())
    tracker._command = SimpleNamespace(
        locked=True, tracking=True, source="forward", velocity_ned=[1.0, 0.0, 0.0]
    )
    tracker._gravity_on = True
    tracker._startup_done = True
    tracker.recorder = SimpleNamespace(observe=lambda *args: reason)
    tracker.telemetry = telemetry
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

    def update(self, t, height_m, attitude, velocity_ned, forward_image, upward_image):
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
    tracker._send = lambda command, height: None
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
