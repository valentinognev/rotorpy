"""Release-gate tests for the offline reliability harness (plan Phase 9, lines
~379-388): no permanent loss after a recoverable camera dropout, no normal homing
on expired observations, no non-finite commands, guidance independent of target
truth, and every trial ends with a defined outcome.
"""

import numpy as np
import pytest

from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.setpoint_stream import PROPOSED_SETPOINT_EXPIRY_S
from rotorpy.guidance.tracking import TrackingState
from rotorpy.guidance.validation.faults import FaultSchedule, FaultWindow
from rotorpy.guidance.validation.harness import TrialSpec, run_trial
from rotorpy.guidance.validation.harness_config import harness_config
from rotorpy.guidance.validation.target_world import TargetMotionProfile


@pytest.fixture(scope="module")
def gate_config():
    return harness_config()


def _stationary_profile():
    return TargetMotionProfile(kind="stationary", start_ground_ned=np.array([100.0, 50.0, 0.0]))


def test_no_permanent_loss_after_a_recoverable_camera_dropout(gate_config):
    """A dropout shorter than `tracking_thresholds()['max_prediction_gap_s']` (2.0 s)
    must not abort the track: once frames resume, the state machine returns to
    `TRACK`, not stuck in `TEMPORARY_LOSS`/`REACQUIRE`/`ABORT`."""
    ticks = []
    schedule = FaultSchedule(camera_dropout=(FaultWindow(2.0, 3.0),))
    spec = TrialSpec(seed=1, target_profile=_stationary_profile(), fault_schedule=schedule, on_tick=ticks.append)
    result = run_trial(spec, config=gate_config)

    locked_before = any(r.tracking_state is TrackingState.TRACK and r.t_s < 2.0 for r in ticks)
    assert locked_before, "test setup: the target must already be locked before the dropout"
    assert not any(r.tracking_state is TrackingState.ABORT for r in ticks)
    recovered = any(r.tracking_state is TrackingState.TRACK for r in ticks if r.t_s > 3.2)
    assert recovered
    assert result.outcome != RunOutcome.TARGET_LOST


def test_no_normal_homing_on_expired_observations(gate_config):
    """Once the held command's observation age exceeds `SetpointStream`'s own
    `expiry_s` (0.5 s), the stream must stop sending the stale velocity -- a longer
    dropout must show a stretch of `transmitted is None`, not a resend."""
    ticks = []
    dropout_start, dropout_end = 2.0, 3.0
    schedule = FaultSchedule(camera_dropout=(FaultWindow(dropout_start, dropout_end),))
    spec = TrialSpec(seed=1, target_profile=_stationary_profile(), fault_schedule=schedule, on_tick=ticks.append)
    run_trial(spec, config=gate_config)

    expired_window_start = dropout_start + PROPOSED_SETPOINT_EXPIRY_S + 0.05
    expired_ticks = [r for r in ticks if expired_window_start <= r.t_s < dropout_end]
    assert expired_ticks, "test setup: the dropout must outlast the setpoint expiry"
    assert all(r.transmitted is None for r in expired_ticks)


def test_no_non_finite_commands_under_any_fault_injection(gate_config):
    schedules = [
        FaultSchedule(),
        FaultSchedule(camera_dropout=(FaultWindow(2.0, 3.0),)),
        FaultSchedule(processing_stall=(FaultWindow(2.0, 2.5),)),
        FaultSchedule(stale_telemetry=(FaultWindow(2.0, 2.5),)),
        FaultSchedule(pause_reset=(2.0,)),
        FaultSchedule(recording_failure=True),
    ]
    for schedule in schedules:
        spec = TrialSpec(seed=1, target_profile=_stationary_profile(), fault_schedule=schedule)
        result = run_trial(spec, config=gate_config)
        assert not result.non_finite_detected, schedule


def test_pause_reset_discontinuity_does_not_abort_the_track(gate_config):
    ticks = []
    schedule = FaultSchedule(pause_reset=(4.0,))
    spec = TrialSpec(seed=1, target_profile=_stationary_profile(), fault_schedule=schedule, on_tick=ticks.append)
    result = run_trial(spec, config=gate_config)
    assert not any(r.tracking_state is TrackingState.ABORT for r in ticks)
    assert result.outcome != RunOutcome.TARGET_LOST


def test_every_trial_ends_with_a_defined_run_outcome(gate_config):
    schedules = [
        FaultSchedule(),
        FaultSchedule(camera_dropout=(FaultWindow(2.0, 5.0),)),
        FaultSchedule(recording_failure=True),
    ]
    for schedule in schedules:
        spec = TrialSpec(seed=2, target_profile=_stationary_profile(), fault_schedule=schedule)
        result = run_trial(spec, config=gate_config)
        assert isinstance(result.outcome, RunOutcome)


def test_guidance_update_never_receives_a_target_truth_argument(gate_config):
    """Structural check alongside `tests/test_homing_truth_boundary.py`: the harness's
    own call into `HomingGuidance.update` carries only the vehicle's own simulated
    state and rendered pixels -- parse the call site for any target-truth name."""
    import inspect

    from rotorpy.guidance.validation import harness as harness_module

    source = inspect.getsource(harness_module.run_trial)
    start = source.index("guidance.update(") + len("guidance.update")
    depth = 0
    end = start
    for index in range(start, len(source)):
        if source[index] == "(":
            depth += 1
        elif source[index] == ")":
            depth -= 1
            if depth == 0:
                end = index + 1
                break
    call_text = source[start:end]
    assert "target_position_ned" not in call_text
    assert "target_profile" not in call_text
