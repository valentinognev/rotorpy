"""Attitude and body-rate offboard setpoints from the Gazebo homing tracker, without ROS."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.controllers.px4_cascade.gains import GAINS
from rotorpy.guidance.homing import HomingCommand
from rotorpy.guidance.homing_config import PROPOSED_TELEMETRY_TIMEOUT_S, load_homing_config
from rotorpy.guidance.offboard_command import (
    CommandMode,
    OffboardGains,
    attitude_setpoint,
    offboard_gains,
    rate_setpoint,
)
from rotorpy.guidance.origin_alignment import EstimatorOriginAlignment
from rotorpy.guidance.vehicle_state import VehicleState

# Fixtures and helpers of the existing startup tests, reused as they are.
from test_gazebo_homing_startup import ROOT, _ingest_estimate, _new_tracker, tracker_module  # noqa: F401

_ATTITUDE_MASK = 7  # ROLL|PITCH|YAW rate ignore
_RATE_MASK = 128  # ATTITUDE_IGNORE


def _config():
    return load_homing_config(ROOT / "examples" / "camera_homing.jsonc")


def _command(velocity=(0.0, 0.0, 0.0), yaw=0.0, acceleration=(0.0, 0.0, 0.0)):
    return HomingCommand(velocity, acceleration, np.zeros(3), yaw, True, True, "forward", None, None)


class _Link:
    """A MAVLink stand-in that records both setpoint kinds."""

    def __init__(self):
        self.position_targets = []
        self.attitude_targets = []
        self.target_system = 1
        self.target_component = 1
        self.mav = SimpleNamespace(
            set_position_target_local_ned_send=lambda *args: self.position_targets.append(args),
            set_attitude_target_send=lambda *args: self.attitude_targets.append(args),
        )
        self.recv_match = lambda **kwargs: None


def _angle_tracker(tracker_module, monkeypatch, mode, hover_thrust=0.42):
    tracker = _new_tracker(tracker_module, monkeypatch)
    config = _config()
    gains, _ = offboard_gains(config, {"MPC_THR_HOVER": hover_thrust})
    tracker._command_mode = mode
    tracker._offboard_gains = gains
    tracker._startup_done = True
    tracker._attitude_skips = 0
    tracker._mav = _Link()
    return tracker


def _ingest(tracker, quat_xyzw=(0.0, 0.0, 0.0, 1.0), velocity=(0.0, 0.0, 0.0), age_s=0.0):
    wall_s = time.monotonic() - age_s
    tracker._vehicle_state.ingest_attitude(sim_s=wall_s, wall_s=wall_s, quat_estimator_xyzw=quat_xyzw)
    tracker._vehicle_state.ingest_local_position(
        wall_s=wall_s, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=velocity
    )


def _sent_quat_xyzw(args):
    w, x, y, z = args[4]
    return np.array([x, y, z, w])


def _angle_between(quat_a_xyzw, quat_b_xyzw):
    return (Rotation.from_quat(quat_a_xyzw).inv() * Rotation.from_quat(quat_b_xyzw)).magnitude()


def _constructed_tracker(tracker_module, monkeypatch, link, warnings=None):
    rospy = tracker_module.rospy
    monkeypatch.setattr(rospy, "wait_for_service", lambda name, timeout=None: None, raising=False)
    monkeypatch.setattr(rospy, "Subscriber", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Timer", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Duration", lambda seconds: seconds, raising=False)
    monkeypatch.setattr(rospy, "ServiceProxy", lambda name, _type: (lambda *args: None), raising=False)
    monkeypatch.setattr(
        rospy, "logwarn", lambda message, *args: None if warnings is None else warnings.append(message % args if args else message),
        raising=False,
    )
    monkeypatch.setattr(tracker_module.CameraHomingTracker, "_set_gravity", lambda self, enabled: None)
    monkeypatch.setattr(tracker_module.mavutil, "mavlink_connection", lambda *args, **kwargs: link)
    return tracker_module.CameraHomingTracker(_config(), "hummingbird_0")


def test_default_constructed_tracker_uses_attitude_mode(tracker_module, monkeypatch):
    monkeypatch.setattr(tracker_module, "dump_px4_parameters", lambda connection, names, **kwargs: {"MPC_THR_HOVER": 0.28})
    link = SimpleNamespace(target_system=1, target_component=1, wait_heartbeat=lambda timeout=None: object())
    tracker = _constructed_tracker(tracker_module, monkeypatch, link)
    try:
        assert tracker._command_mode is CommandMode.ATTITUDE
        assert isinstance(tracker._offboard_gains, OffboardGains)
        assert tracker._attitude_skips == 0
    finally:
        tracker.shutdown()


def test_bare_tracker_without_mode_still_sends_velocity(tracker_module):
    tracker = tracker_module.CameraHomingTracker.__new__(tracker_module.CameraHomingTracker)
    tracker._mav = _Link()
    tracker._startup_done = True
    tracker._publish_setpoint(_command(velocity=(1.0, 2.0, 0.0), yaw=0.4))
    assert len(tracker._mav.position_targets) == 1
    assert tracker._mav.position_targets[0][8:11] == (1.0, 2.0, 0.0)
    assert tracker._mav.attitude_targets == []


@pytest.mark.parametrize("mode", [CommandMode.ATTITUDE, CommandMode.RATE])
def test_startup_commands_go_out_as_velocity_in_angle_modes(tracker_module, monkeypatch, mode):
    tracker = _angle_tracker(tracker_module, monkeypatch, mode)
    _ingest(tracker)
    tracker._startup_done = False
    tracker._publish_setpoint(_command(velocity=(0.0, 0.0, -3.0)))  # a homing-type command before the gate opens
    tracker._startup_done = True
    tracker._publish_setpoint(tracker_module.StartupCommand(velocity_ned=(0.0, 0.0, -3.0), yaw_rad=0.3))
    link = tracker._mav
    assert [t[8:11] for t in link.position_targets] == [(0.0, 0.0, -3.0), (0.0, 0.0, -3.0)]
    assert link.attitude_targets == []
    assert tracker._last_transmitted == [0.0, 0.0, -3.0]


def test_velocity_mode_never_sends_attitude(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.VELOCITY)
    _ingest(tracker)
    tracker._publish_setpoint(_command(velocity=(1.0, 0.0, 0.0)))
    assert len(tracker._mav.position_targets) == 1
    assert tracker._mav.attitude_targets == []


def test_first_attitude_target_after_startup_matches_hover(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.ATTITUDE, hover_thrust=0.42)
    current = Rotation.from_euler("ZYX", [0.7, 0.0, 0.0]).as_quat()
    _ingest(tracker, quat_xyzw=current)
    tracker._publish_setpoint(_command(velocity=(0.0, 0.0, 0.0), yaw=0.7))
    assert tracker._mav.position_targets == []
    assert len(tracker._mav.attitude_targets) == 1
    args = tracker._mav.attitude_targets[0]
    assert args[3] == _ATTITUDE_MASK
    assert args[8] == pytest.approx(0.42, abs=0.02)
    assert np.degrees(_angle_between(_sent_quat_xyzw(args), current)) < 1.0
    # The previous velocity transmission must not be reported as the latest one.
    assert tracker._last_transmitted is None
    assert tracker._last_transmitted_attitude["mode"] == "attitude"


def test_attitude_target_is_in_estimator_frame_with_yaw_offset(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.ATTITUDE)
    # The estimator's north axis points along world east.
    alignment = EstimatorOriginAlignment([0.0, 0.0, -30.0], np.pi / 2.0, 0.0)
    tracker._vehicle_state = VehicleState(alignment)
    # The vehicle faces world north, which is estimator heading -90 degrees.
    facing_world_north = Rotation.from_euler("ZYX", [-np.pi / 2.0, 0.0, 0.0]).as_quat()

    # World heading 0 must be sent as estimator heading -90 degrees.
    _ingest(tracker, quat_xyzw=facing_world_north)
    tracker._publish_setpoint(_command(yaw=0.0))
    args = tracker._mav.attitude_targets[-1]
    yaw, pitch, roll = Rotation.from_quat(_sent_quat_xyzw(args)).as_euler("ZYX")
    assert (yaw, pitch, roll) == pytest.approx((-np.pi / 2.0, 0.0, 0.0), abs=1e-6)

    # World north is estimator west: the thrust must lean along estimator -y.
    _ingest(tracker, quat_xyzw=facing_world_north)
    tracker._publish_setpoint(_command(velocity=(5.0, 0.0, 0.0), yaw=0.0))
    thrust_direction = -Rotation.from_quat(_sent_quat_xyzw(tracker._mav.attitude_targets[-1])).apply([0.0, 0.0, 1.0])
    assert thrust_direction[1] < -0.1
    assert abs(thrust_direction[0]) < 1e-6

    # Already flying world north at 5 m/s (estimator velocity 0, -5, 0): no error, level.
    _ingest(tracker, quat_xyzw=facing_world_north, velocity=(0.0, -5.0, 0.0))
    tracker._publish_setpoint(_command(velocity=(5.0, 0.0, 0.0), yaw=0.0))
    _, pitch, roll = Rotation.from_quat(_sent_quat_xyzw(tracker._mav.attitude_targets[-1])).as_euler("ZYX")
    assert (pitch, roll) == pytest.approx((0.0, 0.0), abs=1e-6)


def test_rate_mode_sends_body_rates_with_attitude_ignored(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.RATE)
    _ingest(tracker)
    tracker._publish_setpoint(_command(velocity=(5.0, 0.0, 0.0), yaw=0.0))
    assert tracker._mav.position_targets == []
    args = tracker._mav.attitude_targets[0]
    assert args[3] == _RATE_MASK
    rates = np.array(args[5:8])
    assert np.all(np.isfinite(rates))
    assert rates[1] < 0.0  # nose-down demand
    assert 0.0 < args[8] <= 1.0
    assert tracker._last_transmitted_attitude["mode"] == "rate"
    assert tracker._last_transmitted_attitude["body_rates_frd"] == pytest.approx(rates)


def test_attitude_quaternion_is_sent_scalar_first(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.ATTITUDE)
    _ingest(tracker)
    tracker._publish_setpoint(_command(velocity=(5.0, 0.0, 0.0), yaw=0.3))
    args = tracker._mav.attitude_targets[0]
    stored = tracker._last_transmitted_attitude["quat_xyzw"]
    assert list(args[4]) == pytest.approx([stored[3], stored[0], stored[1], stored[2]])
    assert args[4][0] > 0.9  # a modest tilt keeps w large; xyzw order would put ~0 here


def test_publisher_skips_attitude_target_when_state_is_stale(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.ATTITUDE)
    _ingest(tracker, age_s=PROPOSED_TELEMETRY_TIMEOUT_S + 0.5)
    tracker._publish_setpoint(_command(velocity=(1.0, 0.0, 0.0)))
    assert tracker._mav.attitude_targets == []
    assert tracker._mav.position_targets == []
    assert tracker._attitude_skips == 1


def test_skip_clears_the_reported_transmitted_attitude(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.ATTITUDE)
    _ingest(tracker)
    tracker._publish_setpoint(_command(velocity=(1.0, 0.0, 0.0)))
    assert tracker._last_transmitted_attitude is not None
    tracker._vehicle_state = type(tracker._vehicle_state)(tracker._vehicle_state.alignment)
    tracker._publish_setpoint(_command(velocity=(1.0, 0.0, 0.0)))
    assert tracker._last_transmitted_attitude is None


def test_publisher_skips_when_no_state_was_ever_received(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.ATTITUDE)
    tracker._publish_setpoint(_command())
    assert tracker._mav.attitude_targets == [] and tracker._mav.position_targets == []
    assert tracker._attitude_skips == 1


def test_publisher_skips_when_only_the_velocity_is_stale(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.ATTITUDE)
    _ingest(tracker)
    stale_s = time.monotonic() - PROPOSED_TELEMETRY_TIMEOUT_S - 0.5
    tracker._vehicle_state.ingest_local_position(
        wall_s=stale_s, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0]
    )
    tracker._publish_setpoint(_command())
    assert tracker._mav.attitude_targets == []
    assert tracker._attitude_skips == 1


@pytest.mark.parametrize("mode", [CommandMode.ATTITUDE, CommandMode.RATE])
def test_publisher_skips_non_finite_commands(tracker_module, monkeypatch, mode):
    tracker = _angle_tracker(tracker_module, monkeypatch, mode)
    _ingest(tracker)
    tracker._publish_setpoint(_command(velocity=(np.nan, 0.0, 0.0)))
    tracker._publish_setpoint(_command(yaw=np.inf))
    assert tracker._mav.attitude_targets == []
    assert tracker._attitude_skips == 2


def test_offboard_gains_read_from_px4_with_fallback(tracker_module, monkeypatch):
    requested = []
    link = SimpleNamespace(target_system=1, target_component=1, wait_heartbeat=lambda timeout=None: object())

    def request(system, component, name, index):
        requested.append(name.decode())

    def receive(type=None, blocking=False, **kwargs):
        if requested and requested[-1] == "MPC_THR_HOVER":
            return SimpleNamespace(param_id="MPC_THR_HOVER\x00\x00", param_value=0.42)
        return None

    link.mav = SimpleNamespace(param_request_read_send=request)
    link.recv_match = receive
    real_dump = tracker_module.dump_px4_parameters
    monkeypatch.setattr(
        tracker_module, "dump_px4_parameters", lambda connection, names: real_dump(connection, names, timeout_s=0.01)
    )
    warnings = []
    tracker = _constructed_tracker(tracker_module, monkeypatch, link, warnings)
    try:
        gains = tracker._offboard_gains
        assert gains.hover_thrust == 0.42
        assert gains.roll_p == GAINS["MC_ROLL_P"]
        assert gains.thr_max == GAINS["MPC_THR_MAX"]
        assert gains.tilt_max_rad == pytest.approx(np.deg2rad(GAINS["MPC_TILTMAX_AIR"]))
        assert set(requested) == set(tracker_module.OFFBOARD_PX4_PARAMETERS)
        fallbacks = [name for name in tracker_module.OFFBOARD_PX4_PARAMETERS if name != "MPC_THR_HOVER"]
        for name in fallbacks:
            assert sum(name in message for message in warnings) == 1, name
        assert not any("MPC_THR_HOVER" in message for message in warnings)
    finally:
        tracker.shutdown()


def _scripted_dump(monkeypatch, tracker_module, replies):
    """Patch the parameter read: each call pops the next reply (an exception is raised)."""
    calls = []

    def dump(connection, names, **kwargs):
        calls.append(tuple(names))
        reply = replies.pop(0) if replies else {}
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(tracker_module, "dump_px4_parameters", dump)
    return calls


def test_hover_thrust_retried_alone_when_first_read_misses_it(tracker_module, monkeypatch):
    others = {name: GAINS[name] for name in tracker_module.OFFBOARD_PX4_PARAMETERS if name != "MPC_THR_HOVER"}
    calls = _scripted_dump(monkeypatch, tracker_module, [others, {}, {"MPC_THR_HOVER": 0.28}])
    link = SimpleNamespace(target_system=1, target_component=1, wait_heartbeat=lambda timeout=None: object())
    warnings = []
    tracker = _constructed_tracker(tracker_module, monkeypatch, link, warnings)
    try:
        assert tracker._offboard_gains.hover_thrust == 0.28
        assert tracker._command_mode is CommandMode.ATTITUDE
        assert calls[1:] == [("MPC_THR_HOVER",)] * 2
        assert not any("MPC_THR_HOVER" in message for message in warnings)
    finally:
        tracker.shutdown()


@pytest.mark.parametrize("failure", [{}, OSError("link down")])
def test_unknown_hover_thrust_disables_angle_commands(tracker_module, monkeypatch, failure):
    calls = _scripted_dump(monkeypatch, tracker_module, [failure] * 4)
    link = SimpleNamespace(target_system=1, target_component=1, wait_heartbeat=lambda timeout=None: object())
    warnings = []
    tracker = _constructed_tracker(tracker_module, monkeypatch, link, warnings)
    try:
        assert len(calls) == 4  # the full read plus three hover-only retries
        assert calls[1:] == [("MPC_THR_HOVER",)] * 3
        assert tracker._command_mode is CommandMode.VELOCITY
        assert any("hover thrust is unknown" in message and "disabled" in message for message in warnings)
        tracker._mav = _Link()
        tracker._startup_done = True
        _ingest(tracker)
        tracker._publish_setpoint(_command(velocity=(1.0, 2.0, 0.0), yaw=0.4))
        assert len(tracker._mav.position_targets) == 1
        assert tracker._mav.position_targets[0][8:11] == (1.0, 2.0, 0.0)
        assert tracker._mav.attitude_targets == []
    finally:
        tracker.shutdown()


def test_rate_mode_rates_follow_estimator_frame_with_yaw_offset(tracker_module, monkeypatch):
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.RATE)
    tracker._vehicle_state = VehicleState(EstimatorOriginAlignment([0.0, 0.0, -30.0], np.pi / 2.0, 0.0))
    # Facing world north is estimator heading -90 degrees; add a 0.2 rad roll error.
    current_est = Rotation.from_euler("ZYX", [-np.pi / 2.0, 0.0, 0.2]).as_quat()
    _ingest(tracker, quat_xyzw=current_est)
    tracker._publish_setpoint(_command(velocity=(5.0, 0.0, 0.0), yaw=0.0))

    # World north is estimator west (0, -5, 0) and world heading 0 is estimator -90 degrees.
    gains = tracker._offboard_gains
    expected = rate_setpoint(
        attitude_setpoint([0.0, -5.0, 0.0], np.zeros(3), -np.pi / 2.0, np.zeros(3), gains), current_est, gains
    ).body_rates_frd
    sent = np.array(tracker._mav.attitude_targets[0][5:8])
    assert sent == pytest.approx(expected, abs=1e-9)
    assert sent[0] < 0.0  # roll back towards level
    assert sent[1] < 0.0  # nose-down demand


def test_unreadable_px4_parameters_fall_back_without_failing_startup(tracker_module, monkeypatch):
    class BrokenLink(SimpleNamespace):
        @property
        def mav(self):
            raise OSError("link down")

    link = BrokenLink(target_system=1, target_component=1, wait_heartbeat=lambda timeout=None: object())
    warnings = []
    tracker = _constructed_tracker(tracker_module, monkeypatch, link, warnings)
    try:
        assert tracker._startup_error is None
        assert tracker._command_mode is CommandMode.VELOCITY
        assert warnings
    finally:
        tracker.shutdown()


def test_velocity_mode_does_not_query_px4_parameters(tracker_module, monkeypatch):
    calls = []
    monkeypatch.setattr(tracker_module, "dump_px4_parameters", lambda *args, **kwargs: calls.append(args) or {})
    link = SimpleNamespace(target_system=1, target_component=1, wait_heartbeat=lambda timeout=None: object())
    config = dict(_config(), command_mode="velocity")
    rospy = tracker_module.rospy
    monkeypatch.setattr(rospy, "wait_for_service", lambda name, timeout=None: None, raising=False)
    monkeypatch.setattr(rospy, "Subscriber", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Timer", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(rospy, "Duration", lambda seconds: seconds, raising=False)
    monkeypatch.setattr(rospy, "ServiceProxy", lambda name, _type: (lambda *args: None), raising=False)
    monkeypatch.setattr(tracker_module.CameraHomingTracker, "_set_gravity", lambda self, enabled: None)
    monkeypatch.setattr(tracker_module.mavutil, "mavlink_connection", lambda *args, **kwargs: link)
    tracker = tracker_module.CameraHomingTracker(config, "hummingbird_0")
    try:
        assert tracker._command_mode is CommandMode.VELOCITY
        assert calls == []
    finally:
        tracker.shutdown()


def test_vehicle_state_exposes_its_alignment():
    alignment = EstimatorOriginAlignment([0.0, 0.0, -30.0], 0.5, 0.0)
    assert VehicleState(alignment).alignment is alignment


def test_attitude_send_is_serialized_with_mav_lock(tracker_module, monkeypatch):
    # pymavlink is not thread-safe: the publisher's attitude send and the step thread's
    # `_poll_mavlink` must never run inside the same connection at once.
    tracker = _angle_tracker(tracker_module, monkeypatch, CommandMode.ATTITUDE)
    tracker._hb_armed = False
    tracker._hb_mode = None
    tracker._last_status = None
    _ingest(tracker)

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

    tracker._mav.mav.set_attitude_target_send = slow_send
    tracker._mav.recv_match = slow_recv
    command = _command(velocity=(1.0, 0.0, 0.0))
    send_thread = threading.Thread(target=lambda: tracker._publish_setpoint(command))
    poll_thread = threading.Thread(target=tracker._poll_mavlink)
    send_thread.start()
    poll_thread.start()
    send_thread.join()
    poll_thread.join()
    assert overlap == []
    assert tracker._last_transmitted_attitude is not None


def _recorded_tick(tracker_module, monkeypatch, **state):
    from test_gazebo_homing_startup import _FakeGuidance, _fresh_estimate

    seen = []
    tracker = _new_tracker(tracker_module, monkeypatch)
    tracker.telemetry = SimpleNamespace(step=lambda **kwargs: seen.append(kwargs) or True)
    tracker._motors_ready = lambda: False
    tracker.guidance = _FakeGuidance()
    for name, value in state.items():
        setattr(tracker, name, value)
    _fresh_estimate(tracker)
    tracker.step()
    return seen[-1]


def test_telemetry_tick_carries_the_transmitted_attitude_command(tracker_module, monkeypatch):
    tick = _recorded_tick(
        tracker_module, monkeypatch,
        _command_mode=CommandMode.ATTITUDE,
        _last_transmitted=None,
        _last_transmitted_attitude={
            "mode": "attitude",
            "quat_xyzw": np.array([0.0, 0.1, 0.0, 0.995]),
            "thrust": np.float64(0.42),
            "body_rates_frd": None,
        },
    )
    assert tick["transmitted_setpoint"] is None
    assert tick["command_mode"] == "attitude"
    assert tick["transmitted_attitude_quat"] == [0.0, 0.1, 0.0, 0.995]
    assert tick["transmitted_thrust"] == 0.42
    assert tick["transmitted_body_rates_frd"] is None
    assert type(tick["transmitted_thrust"]) is float


def test_telemetry_tick_carries_transmitted_body_rates_in_rate_mode(tracker_module, monkeypatch):
    tick = _recorded_tick(
        tracker_module, monkeypatch,
        _command_mode=CommandMode.RATE,
        _last_transmitted=None,
        _last_transmitted_attitude={
            "mode": "rate", "quat_xyzw": [0.0, 0.0, 0.0, 1.0], "thrust": 0.5,
            "body_rates_frd": np.array([0.1, -0.2, 0.0]),
        },
    )
    assert tick["command_mode"] == "rate"
    assert tick["transmitted_body_rates_frd"] == [0.1, -0.2, 0.0]


def test_telemetry_tick_reports_velocity_when_only_a_velocity_was_sent(tracker_module, monkeypatch):
    tick = _recorded_tick(
        tracker_module, monkeypatch,
        _command_mode=CommandMode.ATTITUDE,
        _last_transmitted=[0.0, 0.0, -3.0],
        _last_transmitted_attitude=None,
    )
    assert tick["command_mode"] == "velocity"
    assert tick["transmitted_setpoint"] == [0.0, 0.0, -3.0]
    assert tick.get("transmitted_attitude_quat") is None
    assert tick.get("transmitted_thrust") is None
    assert tick.get("transmitted_body_rates_frd") is None