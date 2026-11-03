"""Truth-side scoring: target contact vs ground contact, moving target, between samples."""

import numpy as np
import pytest

from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.scoring import (
    TargetShape,
    TraceBuffer,
    TruthScorer,
    distance_to_cylinder,
    score_traces,
    segment_closest_approach,
)

SHAPE = TargetShape(radius_m=0.75, height_m=3.0)


def _scorer(**kwargs):
    defaults = dict(shape=SHAPE, timeout_s=20.0, contact_margin_m=0.3, reach_radius_m=1.5)
    defaults.update(kwargs)
    return TruthScorer(**defaults)


def test_distance_to_a_finite_vertical_cylinder_in_ned():
    # Relative to the ground contact point; the cylinder fills z in [-3, 0] (up is negative).
    assert distance_to_cylinder([0.0, 0.0, -1.5], SHAPE) == 0.0
    assert distance_to_cylinder([2.0, 0.0, -1.5], SHAPE) == pytest.approx(1.25)
    assert distance_to_cylinder([0.0, 0.0, -5.0], SHAPE) == pytest.approx(2.0)
    assert distance_to_cylinder([0.0, 0.0, 1.0], SHAPE) == pytest.approx(1.0)
    assert distance_to_cylinder([1.75, 0.0, -4.0], SHAPE) == pytest.approx(np.hypot(1.0, 1.0))


def test_segment_through_the_cylinder_between_samples_is_a_contact():
    # Both endpoints are 20 m from the target: a sample-only check would see no contact.
    distance, fraction = segment_closest_approach([-20.0, 0.0, -1.5], [20.0, 0.0, -1.5], SHAPE)
    assert distance == 0.0
    assert fraction == pytest.approx(0.5, abs=0.03)


def test_segment_passing_beside_the_cylinder_reports_the_clearance():
    distance, _ = segment_closest_approach([-20.0, 3.0, -1.5], [20.0, 3.0, -1.5], SHAPE)
    assert distance == pytest.approx(2.25, abs=1e-6)


def test_trace_buffer_interpolates_linearly_and_is_bounded():
    trace = TraceBuffer(maxlen=3)
    for t in range(5):
        trace.append(float(t), [float(t), 0.0, 0.0])
    assert len(trace) == 3
    assert trace.at(3.5, max_gap_s=1.0) == pytest.approx([3.5, 0.0, 0.0])
    assert trace.at(0.5, max_gap_s=0.1) is None  # dropped by the bound
    assert trace.at(10.0, max_gap_s=1.0) is None  # too far past the newest sample


def test_trace_buffer_rejects_non_monotonic_and_non_finite_samples():
    trace = TraceBuffer(maxlen=8)
    assert trace.append(1.0, [0.0, 0.0, 0.0])
    assert not trace.append(1.0, [1.0, 0.0, 0.0])
    assert not trace.append(0.5, [1.0, 0.0, 0.0])
    assert not trace.append(2.0, [np.nan, 0.0, 0.0])
    assert trace.rejected == 3
    assert len(trace) == 1


def test_moving_target_is_scored_at_matching_timestamps():
    # The target drives north at 10 m/s; the vehicle samples are offset in time from the
    # target samples. Matching by index (or by latest sample) would give a 5 m miss;
    # interpolating both to common times gives a pass at 1 m.
    scorer = _scorer(contact_margin_m=0.1)
    for t in (0.0, 1.0, 2.0, 3.0):
        scorer.add_target_sample(t, [10.0 * t, 0.0, 0.0])
    for t, north in ((0.25, 2.5), (1.75, 17.5)):
        scorer.add_vehicle_sample(t, [north, 1.0 + 0.75, -1.5])
    result = scorer.result()
    assert result.min_surface_distance_m == pytest.approx(1.0, abs=1e-6)
    assert not result.target_contact
    assert result.matched_samples == 2


def test_high_speed_pass_between_samples_is_target_contact():
    scorer = _scorer()
    scorer.add_target_sample(0.0, [100.0, 50.0, 0.0])
    scorer.add_target_sample(2.0, [100.0, 50.0, 0.0])
    assert scorer.add_vehicle_sample(0.0, [80.0, 50.0, -1.5]) is None
    status = scorer.add_vehicle_sample(1.0, [120.0, 50.0, -1.5])
    assert status == "target"
    result = scorer.result()
    assert result.target_contact
    assert 0.0 < result.contact_time_s < 1.0
    assert scorer.outcome() is RunOutcome.TARGET_REACHED


def test_target_contact_and_ground_contact_are_separate_results():
    scorer = _scorer()
    scorer.add_target_sample(0.0, [100.0, 50.0, 0.0])
    scorer.add_vehicle_sample(0.0, [0.0, 0.0, -30.0])
    assert scorer.add_vehicle_sample(1.0, [10.0, 0.0, -0.1]) == "ground"
    result = scorer.result()
    assert result.ground_contact and not result.target_contact
    assert scorer.outcome() is RunOutcome.GROUND_MISS


def test_ground_contact_beside_the_target_within_reach_radius_scores_reached():
    scorer = _scorer()
    scorer.add_target_sample(0.0, [100.0, 50.0, 0.0])
    scorer.add_vehicle_sample(0.0, [90.0, 50.0, -10.0])
    assert scorer.add_vehicle_sample(1.0, [100.0, 51.5, -0.1]) == "ground"
    assert scorer.result().min_surface_distance_m < 1.5
    assert scorer.outcome() is RunOutcome.TARGET_REACHED


def test_ground_contact_alone_is_never_target_reached():
    scorer = _scorer()
    scorer.add_target_sample(0.0, [100.0, 50.0, 0.0])
    scorer.add_vehicle_sample(0.0, [0.0, 0.0, -30.0])
    scorer.add_vehicle_sample(1.0, [0.0, 0.0, -0.1])
    assert scorer.outcome() is RunOutcome.GROUND_MISS


def test_timeout_and_unfinished_runs():
    scorer = _scorer(timeout_s=2.0)
    scorer.add_target_sample(0.0, [100.0, 50.0, 0.0])
    assert scorer.add_vehicle_sample(5.0, [0.0, 0.0, -30.0]) is None  # elapsed 0 at the first sample
    assert scorer.outcome() is None
    assert scorer.add_vehicle_sample(7.5, [0.0, 0.0, -30.0]) == "timeout"
    assert scorer.outcome() is RunOutcome.TIMEOUT


def test_collision_events_report_contact_even_without_a_sample_near_it():
    scorer = _scorer()
    scorer.add_target_sample(0.0, [100.0, 50.0, 0.0])
    scorer.add_vehicle_sample(0.0, [0.0, 0.0, -30.0])
    assert scorer.note_collision(0.7, "target") == "target"
    assert scorer.result().target_contact
    assert scorer.result().contact_time_s == pytest.approx(0.7)
    assert scorer.outcome() is RunOutcome.TARGET_REACHED


def test_a_ground_collision_event_does_not_score_target_reached():
    scorer = _scorer()
    scorer.add_target_sample(0.0, [100.0, 50.0, 0.0])
    scorer.add_vehicle_sample(0.0, [0.0, 0.0, -30.0])
    assert scorer.note_collision(0.7, "ground") == "ground"
    assert scorer.outcome() is RunOutcome.GROUND_MISS


def test_unknown_collision_kinds_are_ignored():
    scorer = _scorer()
    assert scorer.note_collision(0.7, "ground_plane_of_another_model") is None


def test_static_fallback_target_is_used_when_no_target_trace_exists():
    scorer = _scorer(static_target_ground_ned=[100.0, 50.0, 0.0])
    scorer.add_vehicle_sample(0.0, [90.0, 50.0, -1.5])
    status = scorer.add_vehicle_sample(1.0, [110.0, 50.0, -1.5])
    assert status == "target"


def test_stale_target_trace_is_counted_not_scored():
    scorer = _scorer(max_target_gap_s=0.5)
    scorer.add_target_sample(0.0, [100.0, 50.0, 0.0])
    scorer.add_vehicle_sample(10.0, [100.0, 50.0, -1.5])
    result = scorer.result()
    assert result.matched_samples == 0
    assert result.unmatched_samples == 1
    assert not result.target_contact


def test_non_finite_vehicle_samples_never_score():
    scorer = _scorer(static_target_ground_ned=[100.0, 50.0, 0.0])
    assert scorer.add_vehicle_sample(0.0, [np.nan, 0.0, 0.0]) is None
    assert scorer.result().samples == 0


def test_batch_scoring_matches_the_online_scorer():
    target = [(0.0, [100.0, 50.0, 0.0]), (2.0, [100.0, 50.0, 0.0])]
    vehicle = [(0.0, [80.0, 50.0, -1.5]), (1.0, [120.0, 50.0, -1.5])]
    result = score_traces(vehicle, target, SHAPE, contact_margin_m=0.3)
    assert result.target_contact
    assert result.min_surface_distance_m == 0.0
