"""Unit tests for the reliability-campaign statistics and CLI (plan Phase 9): Wilson
confidence intervals, aggregated metrics, and a `--quick` smoke run. The real
>=100-nominal-trial campaign is never run inside this suite -- see
`task-9-report.md` for the one, separately-run, manual campaign invocation."""

import math

import pytest

from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.validation.campaign import DISCLAIMER, format_report, main, run_campaign
from rotorpy.guidance.validation.harness import TrialResult
from rotorpy.guidance.validation.stats import aggregate_metrics, wilson_interval


def test_wilson_interval_matches_a_hand_computed_value():
    # n=100, successes=95: standard Wilson interval, z=1.96.
    low, high = wilson_interval(95, 100)
    assert low == pytest.approx(0.8882, abs=2e-3)
    assert high == pytest.approx(0.9785, abs=2e-3)


def test_wilson_interval_handles_zero_trials_without_dividing_by_zero():
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_is_symmetric_about_0_5_at_a_coin_flip():
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(1.0 - high, abs=1e-9)


def test_wilson_interval_rejects_impossible_counts():
    with pytest.raises(ValueError):
        wilson_interval(5, 3)
    with pytest.raises(ValueError):
        wilson_interval(-1, 3)


def _result(outcome, **overrides):
    defaults = dict(
        outcome=outcome, reason=None, elapsed_s=1.0, miss_m=None, mean_speed_m_s=5.0, max_speed_m_s=8.0,
        mean_tracking_error_m=2.0, visibility_loss_ticks=0, visibility_loss_episodes=0,
        mean_observation_age_s=0.05, max_observation_age_s=0.1, saturation_ticks=0, ticks=100,
        non_finite_detected=False,
    )
    defaults.update(overrides)
    return TrialResult(**defaults)


def test_aggregate_metrics_reports_success_rate_and_failure_categories_together():
    results = [
        _result(RunOutcome.TARGET_REACHED),
        _result(RunOutcome.TARGET_REACHED),
        _result(RunOutcome.TIMEOUT),
        _result(RunOutcome.TARGET_LOST, visibility_loss_episodes=1, visibility_loss_ticks=5),
    ]
    report = aggregate_metrics("nominal", results)
    assert report.n_trials == 4
    assert report.successes == 2
    assert report.success_rate == pytest.approx(0.5)
    assert report.failure_categories == {"TIMEOUT": 1, "TARGET_LOST": 1}
    assert report.visibility_loss_trials == 1
    assert 0.0 <= report.wilson_low <= report.success_rate <= report.wilson_high <= 1.0
    assert report.mean_speed_m_s is not None
    assert report.mean_tracking_error_m is not None


def test_aggregate_metrics_surfaces_any_non_finite_command_across_the_batch():
    results = [_result(RunOutcome.TARGET_REACHED), _result(RunOutcome.SIMULATOR_FAILURE, non_finite_detected=True)]
    report = aggregate_metrics("stress", results)
    assert report.non_finite_detected is True


def test_quick_campaign_runs_a_small_trial_count_and_reports_every_required_field():
    nominal_report, stress_report = run_campaign(seed=0, quick=True)
    assert nominal_report.n_trials <= 4
    assert stress_report.n_trials >= 1
    for report in (nominal_report, stress_report):
        text = format_report(report)
        assert "success_rate" in text
        assert "wilson95%" in text
        assert "speed_m_s" in text
        assert "tracking_error_mean_m" in text
        assert "visibility_loss_trials" in text
        assert "observation_age_s" in text
        assert "saturation_trials" in text
        assert "failure_categories" in text


def test_campaign_disclaimer_states_simulation_model_not_gazebo_validation():
    assert "NOT GAZEBO VALIDATION" in DISCLAIMER
    assert "has NOT been performed" in DISCLAIMER


def test_cli_quick_mode_exits_zero_and_prints_the_disclaimer(capsys):
    exit_code = main(["--quick"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "NOT GAZEBO VALIDATION" in out
    assert "[nominal]" in out
    assert "[stress]" in out
