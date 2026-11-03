import numpy as np
from rotorpy.guidance.outcome import (
    RunOutcome,
    classify_homing_outcome,
    classify_run,
    miss_outcome,
    timeout_outcome,
)

def test_ground_point_is_half_a_meter_from_the_volume_center():
    outcome = miss_outcome(np.array([100.0, 50.0, 0.0]), 12.0)
    assert outcome.miss_m == 0.5
    assert outcome.timed_out is False
    assert outcome.time_s == 12.0

def test_timeout_has_no_miss():
    outcome = timeout_outcome(20.0)
    assert outcome.miss_m is None
    assert outcome.timed_out is True
    assert outcome.time_s == 20.0


def test_outcome_names():
    assert {outcome.name for outcome in RunOutcome} == {
        "TARGET_REACHED", "GROUND_MISS", "TIMEOUT", "TARGET_LOST",
        "ESTIMATOR_FAILURE", "OFFBOARD_FAILURE", "SIMULATOR_FAILURE", "STARTUP_FAILURE",
    }


def test_ground_contact_alone_is_never_target_reached():
    assert classify_run(ground_contact=True) is RunOutcome.GROUND_MISS
    assert classify_run(ground_contact=True, miss_m=40.0) is RunOutcome.GROUND_MISS
    assert classify_run(ground_contact=True, miss_m=float("nan")) is RunOutcome.GROUND_MISS


def test_target_reached_needs_a_miss_inside_the_reach_radius():
    assert classify_run(ground_contact=True, miss_m=0.5, reach_radius_m=1.0) is RunOutcome.TARGET_REACHED
    assert classify_run(ground_contact=True, miss_m=1.0, reach_radius_m=1.0) is RunOutcome.TARGET_REACHED
    assert classify_run(ground_contact=True, miss_m=1.01, reach_radius_m=1.0) is RunOutcome.GROUND_MISS
    assert classify_run(miss_m=0.5, reach_radius_m=1.0) is RunOutcome.TARGET_REACHED


def test_timeout_and_target_lost():
    assert classify_run(timed_out=True) is RunOutcome.TIMEOUT
    assert classify_run(timed_out=True, target_lost=True) is RunOutcome.TARGET_LOST
    assert classify_run(target_lost=True) is RunOutcome.TARGET_LOST


def test_failures_take_precedence_in_a_fixed_order():
    everything = dict(
        ground_contact=True, timed_out=True, target_lost=True, miss_m=0.1,
        estimator_failed=True, offboard_failed=True, simulator_failed=True, startup_failed=True,
    )
    assert classify_run(**everything) is RunOutcome.STARTUP_FAILURE
    everything["startup_failed"] = False
    assert classify_run(**everything) is RunOutcome.SIMULATOR_FAILURE
    everything["simulator_failed"] = False
    assert classify_run(**everything) is RunOutcome.OFFBOARD_FAILURE
    everything["offboard_failed"] = False
    assert classify_run(**everything) is RunOutcome.ESTIMATOR_FAILURE
    everything["estimator_failed"] = False
    assert classify_run(**everything) is RunOutcome.TARGET_REACHED


def test_unfinished_run_has_no_outcome():
    assert classify_run() is None
    assert classify_run(miss_m=50.0) is None


def test_existing_outcomes_map_to_run_outcomes():
    near = miss_outcome(np.array([100.0, 50.0, 0.0]), 12.0)
    assert classify_homing_outcome(near, reach_radius_m=1.0) is RunOutcome.TARGET_REACHED
    assert classify_homing_outcome(near, reach_radius_m=0.1) is RunOutcome.GROUND_MISS
    assert classify_homing_outcome(timeout_outcome(20.0)) is RunOutcome.TIMEOUT
    far = miss_outcome(np.array([0.0, 0.0, 0.0]), 12.0)
    assert classify_homing_outcome(far) is RunOutcome.GROUND_MISS
