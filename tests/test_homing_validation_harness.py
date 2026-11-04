"""Closed-loop trial harness tests (plan Phase 9): the real guidance stack driven by
the synthetic vehicle/camera, run end to end."""

import numpy as np
import pytest

from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.validation.faults import FaultSchedule
from rotorpy.guidance.validation.harness import TrialSpec, run_trial
from rotorpy.guidance.validation.harness_config import harness_config
from rotorpy.guidance.validation.target_world import TargetMotionProfile

_NON_FAILURE_OUTCOMES = (RunOutcome.TARGET_REACHED, RunOutcome.TIMEOUT, RunOutcome.GROUND_MISS)


@pytest.fixture(scope="module")
def config():
    return harness_config()


def _stationary_spec(seed=0, **overrides):
    profile = TargetMotionProfile(kind="stationary", start_ground_ned=np.array([100.0, 50.0, 0.0]))
    kwargs = dict(seed=seed, target_profile=profile)
    kwargs.update(overrides)
    return TrialSpec(**kwargs)


def test_stationary_trial_ends_with_a_flight_outcome_not_an_infrastructure_failure(config):
    result = run_trial(_stationary_spec(), config=config)
    assert result.outcome in _NON_FAILURE_OUTCOMES
    assert result.ticks > 0
    assert not result.non_finite_detected


def test_same_seed_is_fully_deterministic(config):
    first = run_trial(_stationary_spec(seed=7), config=config)
    second = run_trial(_stationary_spec(seed=7), config=config)
    assert first.outcome == second.outcome
    assert first.ticks == second.ticks
    assert first.miss_m == second.miss_m
    assert first.mean_speed_m_s == second.mean_speed_m_s


def test_moving_target_trial_ends_with_a_defined_outcome(config):
    profile = TargetMotionProfile(
        kind="crossing", start_ground_ned=np.array([100.0, 50.0, 0.0]), speed_m_s=2.0, heading_rad=0.0,
    )
    result = run_trial(TrialSpec(seed=1, target_profile=profile), config=config)
    assert isinstance(result.outcome, RunOutcome)
    assert not result.non_finite_detected


def test_trial_result_reports_speed_tracking_error_visibility_and_saturation_together(config):
    result = run_trial(_stationary_spec(seed=3), config=config)
    # Every field the release gate asks be reported together is present (not every
    # one need be non-trivial for a single short stationary trial, but all must
    # exist and be well-typed).
    assert result.mean_speed_m_s >= 0.0
    assert result.max_speed_m_s >= 0.0
    assert result.visibility_loss_ticks >= 0
    assert result.visibility_loss_episodes >= 0
    assert result.saturation_ticks >= 0
    assert isinstance(result.outcome, RunOutcome)


def test_recording_failure_is_isolated_but_the_outcome_is_still_recorded(config):
    spec = _stationary_spec(seed=5, fault_schedule=FaultSchedule(recording_failure=True))
    result = run_trial(spec, config=config)
    assert isinstance(result.outcome, RunOutcome)
    assert len(result.closer_failures) == 1
    assert result.closer_failures[0][0] == "recording"
