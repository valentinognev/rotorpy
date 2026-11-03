"""Verify the Gazebo startup hold and PX4 heartbeat handling without ROS."""

import importlib.util
from pathlib import Path
import sys
import threading
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
    tracker_module.CameraHomingTracker(config, "hummingbird_0")
    assert events.index(("gravity", False)) < events.index(("call", "/gazebo/unpause_physics"))

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
    assert tracker._flight_state()
    assert tracker._flight_state()
    messages.append(SimpleNamespace(get_type=lambda: "HEARTBEAT", type=0, base_mode=0, custom_mode=0))
    assert not tracker._flight_state()


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


def test_gravity_stays_off_until_armed_offboard_motors_spin(tracker_module, monkeypatch):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._lock = threading.Lock()
    tracker._fresh = False
    tracker._forward = tracker._upward = None
    tracker._command = SimpleNamespace(
        locked=False, tracking=False, source=None, velocity_ned=[0.0, 0.0, 0.0]
    )
    tracker._pose = lambda: (np.array([0.0, 0.0, 30.0]), None, np.zeros(3))
    tracker._send = lambda command, height: None
    tracker._sent = 20
    tracker._flight_state = lambda: True
    tracker._gravity_on = False
    tracker._motor_start_time = None
    tracker._startup_done = False
    tracker._startup_height = 30.0
    tracker._hold_start_time = None
    tracker._settled_since = None
    tracker.recorder = None
    tracker._set_gravity = lambda enabled: setattr(tracker, "_gravity_on", enabled)
    monkeypatch.setattr(tracker_module.rospy, "get_time", lambda: 10.0, raising=False)
    tracker._motors_ready = lambda: False
    tracker.step()
    assert not tracker._gravity_on
    tracker._motors_ready = lambda: True
    tracker.step()
    assert tracker._gravity_on
    assert not tracker._startup_done


@pytest.mark.parametrize("reason", ["ground", "timeout"])
def test_tracker_step_reports_flight_end(tracker_module, monkeypatch, reason):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._lock = threading.Lock()
    tracker._fresh = False
    tracker._forward = tracker._upward = None
    tracker._command = SimpleNamespace(
        locked=True, tracking=True, source="forward", velocity_ned=[1.0, 0.0, 0.0]
    )
    tracker._pose = lambda: (np.array([0.0, 0.0, 30.0]), None, np.zeros(3))
    tracker._send = lambda command, height: None
    tracker._sent = 20
    tracker._flight_state = lambda: True
    tracker._gravity_on = True
    tracker._startup_done = True
    tracker.recorder = SimpleNamespace(observe=lambda *args: reason)
    monkeypatch.setattr(tracker_module.rospy, "get_time", lambda: 10.0, raising=False)
    assert tracker.step() == reason


@pytest.mark.parametrize("reason", ["ground", "timeout"])
@pytest.mark.parametrize("has_recording", [True, False])
def test_main_stops_after_saving_flight(tracker_module, monkeypatch, capsys, reason, has_recording):
    events = []
    statuses = iter([None, reason])
    fake_tracker = SimpleNamespace(
        step=lambda: events.append("step") or next(statuses),
        write_page=lambda: events.append("write") or ("flight.html" if has_recording else None),
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
    assert events == ["step", "sleep", "step", "write"]
    output = capsys.readouterr()
    assert f"flight finished: {reason}" in output.out
    assert ("no recording was written" in output.err) is not has_recording


def _stepping_tracker(tracker_module, monkeypatch, reason, telemetry):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._lock = threading.Lock()
    tracker._fresh = False
    tracker._forward = tracker._upward = None
    tracker._forward_raw = tracker._upward_raw = None
    tracker._image_times = {}
    tracker._command = SimpleNamespace(
        locked=True, tracking=True, source="forward", velocity_ned=[1.0, 0.0, 0.0]
    )
    tracker._pose = lambda: (np.array([0.0, 0.0, 30.0]), np.array([0.0, 0.0, 0.0, 1.0]), np.zeros(3))
    tracker._send = lambda command, height: None
    tracker._sent = 20
    tracker._flight_state = lambda: True
    tracker._gravity_on = True
    tracker._startup_done = True
    tracker._hb_mode = None
    tracker._target_ground_ned = np.array([100.0, 50.0, 0.0])
    tracker.recorder = SimpleNamespace(observe=lambda *args: reason)
    tracker.telemetry = telemetry
    monkeypatch.setattr(tracker_module.rospy, "get_time", lambda: 10.0, raising=False)
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
    tracker._forward_raw = tracker._upward_raw = raw
    tracker._forward = tracker._upward = tracker_module.frame_to_float(raw)
    tracker._command = None
    tracker._fresh = True
    tracker._image_times = {"forward": (9.9, 9.95), "upward": (9.9, 9.96)}
    assert tracker.step() == "timeout"
    recorder.log.close()

    run = read_telemetry(recorder.log.path)
    assert run.metadata["rendering_mode"] == "gpu"
    assert run.ticks[0].image_receipt_s == {"forward": 9.95, "upward": 9.96}
    assert run.ticks[0].position_ned == [0.0, 0.0, -30.0]
    assert run.ticks[0].state == "HOMING"
    assert run.outcome["outcome"] == RunOutcome.TIMEOUT.value
    assert run.outcome["reason"] == "timeout"
    assert replay_run(recorder.log.path).matches
