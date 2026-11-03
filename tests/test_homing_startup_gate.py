"""Startup gate: what is sent (or not) before homing, bounded waits, STARTUP_FAILURE."""

import math

import pytest

from rotorpy.guidance.homing_config import startup_thresholds
from rotorpy.guidance.startup import (
    StartupConfig,
    StartupFailure,
    StartupGate,
    StartupInputs,
    StartupMode,
    StartupPhase,
    bounded_service_wait,
    parse_startup_mode,
    startup_velocity,
)

HOLD = 30.0


def _config(mode=StartupMode.TAKEOFF, **overrides):
    values = dict(startup_thresholds(), hold_height_m=HOLD, mode=mode)
    values.update(overrides)
    return StartupConfig(**values)


def _inputs(now, **overrides):
    values = dict(
        now_s=now, estimate_fresh=True, attitude_valid=True, height_m=0.0,
        velocity_ned=(0.0, 0.0, 0.0), armed=False, offboard=False, motors_ready=None,
    )
    values.update(overrides)
    return StartupInputs(**values)


def test_nothing_is_sent_until_the_estimate_is_fresh_and_the_attitude_valid():
    gate = StartupGate(_config(), now_s=0.0)
    for overrides in (
        dict(estimate_fresh=False),
        dict(attitude_valid=False),
        dict(height_m=None),
        dict(velocity_ned=None),
        dict(velocity_ned=(math.nan, 0.0, 0.0)),
    ):
        decision = gate.update(_inputs(0.1, **overrides))
        assert decision.phase is StartupPhase.WAIT_ESTIMATE
        assert decision.setpoint_velocity_ned is None
        assert not decision.request_arm
        assert not decision.homing_active


def test_setpoint_stream_starts_with_the_estimate_not_with_the_first_image():
    # Carry-over: the stream was silent during startup when the estimate preceded the
    # first image. Startup setpoints depend on the estimate only, never on images.
    gate = StartupGate(_config(), now_s=0.0)
    decision = gate.update(_inputs(0.1))
    assert decision.phase is StartupPhase.ENGAGE
    assert decision.setpoint_velocity_ned == (0.0, 0.0, 0.0)
    assert not decision.homing_active


def test_arm_request_waits_for_a_preliminary_stream_then_retries_a_bounded_number_of_times():
    config = _config(pre_engage_stream_s=1.0, arm_retry_period_s=1.0, max_arm_attempts=3,
                     engage_confirm_timeout_s=1.0)
    gate = StartupGate(config, now_s=0.0)
    gate.update(_inputs(0.1))
    assert not gate.update(_inputs(0.9)).request_arm  # still streaming the offboard prelude
    requests = []
    now = 1.2
    while now < 10.0 and gate.phase is StartupPhase.ENGAGE:
        decision = gate.update(_inputs(now))
        if decision.request_arm:
            requests.append(now)
        now += 0.25
    assert len(requests) == 3
    assert all(later - earlier >= 1.0 for earlier, later in zip(requests, requests[1:]))
    assert gate.phase is StartupPhase.FAILED
    assert "offboard_not_confirmed" in gate.failure


def test_offboard_needs_both_armed_and_offboard_to_count_as_confirmed():
    gate = StartupGate(_config(), now_s=0.0)
    gate.update(_inputs(0.1))
    gate.update(_inputs(1.5, armed=True, offboard=False))
    assert gate.phase is StartupPhase.ENGAGE
    gate.update(_inputs(1.6, armed=False, offboard=True))
    assert gate.phase is StartupPhase.ENGAGE
    gate.update(_inputs(1.7, armed=True, offboard=True))
    assert gate.phase is StartupPhase.CLIMB


def _fly_to_stabilize(gate, hold=HOLD):
    gate.update(_inputs(0.1))
    gate.update(_inputs(1.5, armed=True, offboard=True))
    assert gate.phase is StartupPhase.CLIMB
    gate.update(_inputs(2.0, armed=True, offboard=True, height_m=hold - 0.2))
    assert gate.phase is StartupPhase.STABILIZE


def test_takeoff_mode_climbs_then_stabilizes_then_homes_without_touching_gravity():
    gate = StartupGate(_config(StartupMode.TAKEOFF), now_s=0.0)
    decision = gate.update(_inputs(0.1))
    assert decision.gravity_enabled is None
    gate.update(_inputs(1.5, armed=True, offboard=True))
    climbing = gate.update(_inputs(2.0, armed=True, offboard=True, height_m=5.0))
    assert climbing.phase is StartupPhase.CLIMB
    assert climbing.setpoint_velocity_ned[:2] == (0.0, 0.0)
    assert climbing.setpoint_velocity_ned[2] < 0.0  # up
    assert climbing.gravity_enabled is None
    _fly_to_stabilize(StartupGate(_config(StartupMode.TAKEOFF), now_s=0.0))
    gate = StartupGate(_config(StartupMode.TAKEOFF), now_s=0.0)
    _fly_to_stabilize(gate)
    steady = dict(armed=True, offboard=True, height_m=HOLD)
    assert not gate.update(_inputs(2.1, **steady)).homing_active
    assert not gate.update(_inputs(2.9, **steady)).homing_active
    decision = gate.update(_inputs(3.2, **steady))
    assert decision.homing_active and decision.phase is StartupPhase.HOMING
    assert decision.setpoint_velocity_ned is None  # the homing command takes over


@pytest.mark.parametrize("unstable", [
    dict(velocity_ned=(0.0, 0.0, 1.0)),
    dict(velocity_ned=(1.0, 0.0, 0.0)),
    dict(height_m=HOLD + 2.0),
])
def test_stability_resets_the_settle_timer(unstable):
    gate = StartupGate(_config(), now_s=0.0)
    _fly_to_stabilize(gate)
    steady = dict(armed=True, offboard=True, height_m=HOLD)
    gate.update(_inputs(2.1, **steady))
    gate.update(_inputs(2.9, **dict(steady, **unstable)))
    decision = gate.update(_inputs(3.2, **steady))
    assert not decision.homing_active  # the timer restarted at 3.2


def test_airborne_spawn_mode_holds_gravity_off_until_motors_spin():
    gate = StartupGate(_config(StartupMode.AIRBORNE_SPAWN), now_s=0.0)
    decision = gate.update(_inputs(0.1, height_m=HOLD))
    assert decision.gravity_enabled is False
    # Legacy spawn behaviour: while gravity is off the hold climbs at the startup climb speed.
    assert decision.setpoint_velocity_ned == (0.0, 0.0, -_config().climb_speed_m_s)
    gate.update(_inputs(1.5, armed=True, offboard=True, height_m=HOLD))
    assert gate.phase is StartupPhase.MOTOR_START
    waiting = gate.update(_inputs(2.0, armed=True, offboard=True, height_m=HOLD, motors_ready=False))
    assert waiting.gravity_enabled is False
    released = gate.update(_inputs(2.1, armed=True, offboard=True, height_m=HOLD, motors_ready=True))
    assert released.gravity_enabled is True
    assert released.phase is StartupPhase.STABILIZE


def test_gravity_is_not_released_without_a_fresh_estimate():
    gate = StartupGate(_config(StartupMode.AIRBORNE_SPAWN), now_s=0.0)
    gate.update(_inputs(0.1, height_m=HOLD))
    gate.update(_inputs(1.5, armed=True, offboard=True, height_m=HOLD))
    decision = gate.update(_inputs(2.0, armed=True, offboard=True, height_m=HOLD,
                                   motors_ready=True, estimate_fresh=False))
    assert decision.phase is StartupPhase.WAIT_ESTIMATE
    assert decision.gravity_enabled is False


def test_motor_start_timeout_is_a_startup_failure():
    gate = StartupGate(_config(StartupMode.AIRBORNE_SPAWN, motor_start_timeout_s=5.0), now_s=0.0)
    gate.update(_inputs(0.1, height_m=HOLD))
    gate.update(_inputs(1.5, armed=True, offboard=True, height_m=HOLD))
    decision = gate.update(_inputs(7.0, armed=True, offboard=True, height_m=HOLD, motors_ready=False))
    assert decision.phase is StartupPhase.FAILED
    assert decision.failure == "motor_start_timeout"
    assert decision.setpoint_velocity_ned is None


def test_estimator_wait_stabilization_and_climb_are_all_bounded():
    gate = StartupGate(_config(estimate_timeout_s=3.0), now_s=0.0)
    assert gate.update(_inputs(3.5, estimate_fresh=False)).failure == "estimate_timeout"

    gate = StartupGate(_config(climb_timeout_s=4.0), now_s=0.0)
    gate.update(_inputs(0.1))
    gate.update(_inputs(1.5, armed=True, offboard=True))
    assert gate.update(_inputs(6.0, armed=True, offboard=True, height_m=2.0)).failure == "climb_timeout"

    gate = StartupGate(_config(stabilize_timeout_s=4.0), now_s=0.0)
    _fly_to_stabilize(gate)
    unstable = dict(armed=True, offboard=True, height_m=HOLD, velocity_ned=(0.0, 0.0, 2.0))
    assert gate.update(_inputs(7.0, **unstable)).failure == "stabilization_timeout"


def test_total_startup_time_is_bounded_even_if_the_estimate_flaps():
    gate = StartupGate(_config(total_timeout_s=10.0, estimate_timeout_s=100.0), now_s=0.0)
    now = 0.0
    while now < 20.0 and gate.phase is not StartupPhase.FAILED:
        now += 1.0
        gate.update(_inputs(now, estimate_fresh=int(now) % 2 == 0))
    assert gate.phase is StartupPhase.FAILED
    assert gate.failure == "startup_timeout"


def test_failure_is_terminal_and_sends_nothing():
    gate = StartupGate(_config(estimate_timeout_s=1.0), now_s=0.0)
    gate.update(_inputs(2.0, estimate_fresh=False))
    later = gate.update(_inputs(2.5))
    assert later.phase is StartupPhase.FAILED
    assert later.setpoint_velocity_ned is None and not later.request_arm
    assert later.failure == "estimate_timeout"


def test_losing_offboard_before_homing_returns_to_engage_within_the_attempt_bound():
    gate = StartupGate(_config(), now_s=0.0)
    _fly_to_stabilize(gate)
    gate.update(_inputs(2.1, armed=True, offboard=False, height_m=HOLD))
    assert gate.phase is StartupPhase.ENGAGE


def test_losing_offboard_after_gravity_release_is_fatal():
    gate = StartupGate(_config(StartupMode.AIRBORNE_SPAWN), now_s=0.0)
    gate.update(_inputs(0.1, height_m=HOLD))
    gate.update(_inputs(1.5, armed=True, offboard=True, height_m=HOLD))
    gate.update(_inputs(2.0, armed=True, offboard=True, height_m=HOLD, motors_ready=True))
    decision = gate.update(_inputs(2.1, armed=True, offboard=False, height_m=HOLD))
    assert decision.phase is StartupPhase.FAILED
    assert decision.failure == "offboard_lost_after_gravity_release"


def test_offboard_loss_while_homing_is_an_offboard_failure_after_a_grace_period():
    gate = StartupGate(_config(offboard_loss_grace_s=1.0), now_s=0.0)
    _fly_to_stabilize(gate)
    steady = dict(armed=True, offboard=True, height_m=HOLD)
    gate.update(_inputs(2.1, **steady))
    assert gate.update(_inputs(3.5, **steady)).homing_active
    assert gate.update(_inputs(4.0, armed=True, offboard=False, height_m=HOLD)).homing_active
    recovered = gate.update(_inputs(4.5, **steady))
    assert recovered.homing_active and recovered.failure is None
    gate.update(_inputs(5.0, armed=True, offboard=False, height_m=HOLD))
    lost = gate.update(_inputs(6.2, armed=True, offboard=False, height_m=HOLD))
    assert lost.failure == "offboard_lost"
    assert lost.failure_kind == "offboard"
    assert not lost.homing_active


def test_startup_failures_map_to_the_startup_kind():
    gate = StartupGate(_config(estimate_timeout_s=1.0), now_s=0.0)
    assert gate.update(_inputs(2.0, estimate_fresh=False)).failure_kind == "startup"


def test_startup_velocity_policy():
    config = _config()
    assert startup_velocity(StartupMode.TAKEOFF, StartupPhase.ENGAGE, False, 0.0, HOLD, config.climb_speed_m_s) == (0.0, 0.0, 0.0)
    climb = startup_velocity(StartupMode.TAKEOFF, StartupPhase.CLIMB, None, 0.0, HOLD, config.climb_speed_m_s)
    assert climb == (0.0, 0.0, -config.climb_speed_m_s)
    near = startup_velocity(StartupMode.TAKEOFF, StartupPhase.STABILIZE, None, HOLD - 0.5, HOLD, config.climb_speed_m_s)
    assert near == (0.0, 0.0, -0.5)
    over = startup_velocity(StartupMode.AIRBORNE_SPAWN, StartupPhase.STABILIZE, True, HOLD + 5.0, HOLD, config.climb_speed_m_s)
    assert over == (0.0, 0.0, 2.0)
    under = startup_velocity(StartupMode.AIRBORNE_SPAWN, StartupPhase.STABILIZE, True, HOLD - 5.0, HOLD, config.climb_speed_m_s)
    assert under == (0.0, 0.0, -2.0)
    assert startup_velocity(StartupMode.AIRBORNE_SPAWN, StartupPhase.MOTOR_START, False, HOLD, HOLD, 3.0) == (0.0, 0.0, -3.0)
    for phase in (StartupPhase.WAIT_ESTIMATE, StartupPhase.HOMING, StartupPhase.FAILED):
        assert startup_velocity(StartupMode.TAKEOFF, phase, None, 0.0, HOLD, 3.0) is None


def test_mode_parsing_is_explicit():
    assert parse_startup_mode("takeoff") is StartupMode.TAKEOFF
    assert parse_startup_mode("airborne_spawn") is StartupMode.AIRBORNE_SPAWN
    with pytest.raises(ValueError, match="startup mode"):
        parse_startup_mode("teleport")


def test_config_rejects_non_positive_bounds():
    with pytest.raises(ValueError):
        _config(estimate_timeout_s=0.0)
    with pytest.raises(ValueError):
        _config(max_arm_attempts=0)


def test_documented_defaults_exist_and_airborne_spawn_is_the_proposed_legacy_default():
    from rotorpy.guidance import homing_config

    assert homing_config.PROPOSED_STARTUP_MODE == "airborne_spawn"
    assert set(startup_thresholds()) >= {
        "estimate_timeout_s", "max_arm_attempts", "motor_start_timeout_s", "stabilize_timeout_s",
        "climb_timeout_s", "total_timeout_s", "settle_hold_s",
    }


def test_service_waits_are_bounded_and_name_the_missing_service():
    clock = [0.0]
    calls = []

    def wait_one(name, timeout):
        calls.append((name, round(timeout, 3)))
        if name == "/gazebo/b":
            clock[0] += timeout
            raise RuntimeError("timeout")

    with pytest.raises(StartupFailure) as info:
        bounded_service_wait(["/gazebo/a", "/gazebo/b", "/gazebo/c"], wait_one, 10.0, lambda: clock[0])
    assert "service_timeout:/gazebo/b" in str(info.value)
    assert [name for name, _ in calls] == ["/gazebo/a", "/gazebo/b"]
    assert all(timeout <= 10.0 for _, timeout in calls)


def test_service_waits_share_one_total_budget():
    clock = [0.0]

    def wait_one(name, timeout):
        clock[0] += 6.0

    with pytest.raises(StartupFailure, match="service_timeout"):
        bounded_service_wait(["/a", "/b", "/c"], wait_one, 10.0, lambda: clock[0])


def test_service_wait_success_returns_quietly():
    bounded_service_wait(["/a"], lambda name, timeout: None, 5.0, lambda: 0.0)


def test_takeoff_mode_puts_the_estimator_origin_at_the_ground_spawn_not_the_start_height():
    from pathlib import Path

    from rotorpy.guidance.homing_config import (
        PROPOSED_TAKEOFF_SPAWN_HEIGHT_M, alignment_from_config, load_homing_config,
    )

    config = load_homing_config(Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc")
    airborne = alignment_from_config(config)
    assert airborne.origin_world_ned == pytest.approx(config["estimator_origin_world_ned"])
    same_as_default = alignment_from_config(config, startup_mode="airborne_spawn")
    assert same_as_default.origin_world_ned == pytest.approx(airborne.origin_world_ned)
    takeoff = alignment_from_config(config, startup_mode="takeoff")
    origin = config["estimator_origin_world_ned"]
    assert takeoff.origin_world_ned == pytest.approx([origin[0], origin[1], -PROPOSED_TAKEOFF_SPAWN_HEIGHT_M])
    assert takeoff.yaw_offset_rad == airborne.yaw_offset_rad
    # A vehicle at rest on the ground reads about the spawn height above ground, not 30 m.
    height = -takeoff.estimator_to_world([0.0, 0.0, 0.0])[2]
    assert height == pytest.approx(PROPOSED_TAKEOFF_SPAWN_HEIGHT_M)
