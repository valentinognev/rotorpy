"""Unit tests for the pure `tracking.TrackingStateMachine` (no cameras, no geometry).

Defaults chosen here are intentionally small and explicit (not the `homing_config`
proposed defaults) so each behavior is exercised in a handful of ticks.
"""

import pytest

from rotorpy.guidance.tracking import CameraSignal, TrackingState, TrackingStateMachine

SEEN = CameraSignal(visible=True, confidence=1.0)
UNSEEN = CameraSignal(visible=False)


def _machine(acquire_frames=1, reacquire_frames=1, switch_margin=0.1,
             max_prediction_gap_s=1.0, max_uncertainty_px=100.0, camera_priority=()):
    return TrackingStateMachine(
        acquire_frames=acquire_frames,
        reacquire_frames=reacquire_frames,
        switch_margin=switch_margin,
        max_prediction_gap_s=max_prediction_gap_s,
        max_uncertainty_px=max_uncertainty_px,
        camera_priority=camera_priority,
    )


def test_starts_in_startup_with_no_active_camera():
    machine = _machine()
    assert machine.state is TrackingState.STARTUP
    assert machine.camera is None


def test_single_visible_tick_acquires_by_default():
    machine = _machine()
    machine.update(0.0, {"forward": SEEN, "upward": UNSEEN})
    assert machine.state is TrackingState.TRACK
    assert machine.camera == "forward"
    assert machine.discontinuous is True


def test_acquisition_requires_the_configured_consecutive_frame_count():
    machine = _machine(acquire_frames=3)
    machine.update(0.0, {"forward": SEEN})
    assert machine.state is TrackingState.ACQUIRE
    machine.update(0.1, {"forward": SEEN})
    assert machine.state is TrackingState.ACQUIRE
    machine.update(0.2, {"forward": SEEN})
    assert machine.state is TrackingState.TRACK


def test_a_flicker_before_acquisition_does_not_accumulate_permanent_loss():
    # Spec: "Do not accumulate permanent loss during startup."
    machine = _machine(acquire_frames=3, max_prediction_gap_s=0.01)
    machine.update(0.0, {"forward": SEEN})
    assert machine.state is TrackingState.ACQUIRE
    # A long gap with nothing visible, well past max_prediction_gap_s: still not an
    # abort, because acquisition was never confirmed.
    machine.update(10.0, {"forward": UNSEEN})
    assert machine.state is TrackingState.STARTUP
    assert machine.camera is None
    machine.update(10.1, {"forward": SEEN})
    assert machine.state is TrackingState.ACQUIRE


def test_a_long_startup_with_nothing_ever_visible_never_aborts():
    machine = _machine(max_prediction_gap_s=0.01)
    for t in (0.0, 1.0, 2.0, 100.0):
        machine.update(t, {"forward": UNSEEN, "upward": UNSEEN})
    assert machine.state is TrackingState.STARTUP


def test_losing_the_only_camera_enters_temporary_loss_then_recovers():
    machine = _machine(max_prediction_gap_s=5.0)
    machine.update(0.0, {"forward": SEEN})
    assert machine.state is TrackingState.TRACK
    machine.update(0.1, {"forward": UNSEEN})
    assert machine.state is TrackingState.TEMPORARY_LOSS
    assert machine.camera == "forward"  # remembered, not forgotten, through the gap
    machine.update(0.2, {"forward": SEEN})
    assert machine.state is TrackingState.TRACK
    assert machine.discontinuous is True  # a gap occurred: steering must not use stale history


def test_prediction_gap_beyond_the_bound_aborts_with_a_reason():
    machine = _machine(max_prediction_gap_s=1.0)
    machine.update(0.0, {"forward": SEEN})
    machine.update(0.1, {"forward": UNSEEN})
    assert machine.state is TrackingState.TEMPORARY_LOSS
    machine.update(1.5, {"forward": UNSEEN})
    assert machine.state is TrackingState.ABORT
    assert machine.abort_reason == "max_prediction_gap_exceeded"


def test_abort_is_one_way_and_ignores_a_later_sighting():
    machine = _machine(max_prediction_gap_s=1.0)
    machine.update(0.0, {"forward": SEEN})
    machine.update(0.1, {"forward": UNSEEN})
    machine.update(2.0, {"forward": UNSEEN})
    assert machine.state is TrackingState.ABORT
    machine.update(2.1, {"forward": SEEN})
    assert machine.state is TrackingState.ABORT
    assert machine.abort_reason == "max_prediction_gap_exceeded"


def test_growing_track_uncertainty_aborts_before_the_time_gap_does():
    machine = _machine(max_prediction_gap_s=100.0, max_uncertainty_px=10.0)
    machine.update(0.0, {"forward": SEEN})
    machine.update(0.1, {"forward": CameraSignal(visible=False, uncertainty_px=50.0)})
    assert machine.state is TrackingState.ABORT
    assert machine.abort_reason == "max_track_uncertainty_exceeded"


def test_complete_is_external_and_one_way():
    machine = _machine()
    machine.update(0.0, {"forward": SEEN})
    machine.complete("target_reached")
    assert machine.state is TrackingState.COMPLETE
    assert machine.complete_reason == "target_reached"
    machine.update(0.1, {"forward": SEEN})
    assert machine.state is TrackingState.COMPLETE  # one-way: a later sighting cannot undo it


def test_complete_does_not_override_an_existing_abort():
    machine = _machine(max_prediction_gap_s=0.5)
    machine.update(0.0, {"forward": SEEN})
    machine.update(0.1, {"forward": UNSEEN})
    machine.update(1.0, {"forward": UNSEEN})
    assert machine.state is TrackingState.ABORT
    machine.complete("target_reached")
    assert machine.state is TrackingState.ABORT


def test_camera_priority_prefers_the_higher_priority_camera_once_visible():
    machine = _machine(camera_priority=("upward", "forward"))
    machine.update(0.0, {"forward": SEEN, "upward": UNSEEN})
    assert machine.camera == "forward"
    # Upward appears with a lower raw confidence but outranks forward by priority.
    machine.update(0.1, {"forward": SEEN, "upward": CameraSignal(visible=True, confidence=0.1)})
    assert machine.camera == "upward"
    assert machine.discontinuous is True  # a genuine camera handoff while still tracking


def test_losing_the_active_camera_falls_back_immediately_with_no_hysteresis():
    machine = _machine(camera_priority=("upward", "forward"), switch_margin=0.5)
    machine.update(0.0, {"upward": SEEN})
    assert machine.camera == "upward"
    # Upward disappears; forward is visible. No margin protects a camera that is gone.
    machine.update(0.1, {"forward": CameraSignal(visible=True, confidence=0.01)})
    assert machine.camera == "forward"
    assert machine.state is TrackingState.TRACK


def test_switch_margin_protects_the_active_camera_from_a_marginal_rival():
    # Same priority rank (neither named): a rival with only a small confidence lead
    # (inside the margin) must not steal the still-visible active track.
    machine = _machine(switch_margin=0.2)
    machine.update(0.0, {"forward": CameraSignal(visible=True, confidence=0.5)})
    assert machine.camera == "forward"
    machine.update(0.1, {
        "forward": CameraSignal(visible=True, confidence=0.5),
        "upward": CameraSignal(visible=True, confidence=0.6),
    })
    assert machine.camera == "forward"  # 0.6 - 0.5 = 0.1 <= switch_margin (0.2): no steal
    machine.update(0.2, {
        "forward": CameraSignal(visible=True, confidence=0.5),
        "upward": CameraSignal(visible=True, confidence=0.8),
    })
    assert machine.camera == "upward"  # 0.8 - 0.5 = 0.3 > switch_margin: a real lead wins


def test_force_discontinuity_flags_the_next_observation_and_restarts_the_gap_clock():
    machine = _machine(max_prediction_gap_s=1.0)
    machine.update(0.0, {"forward": SEEN})
    machine.update(0.1, {"forward": UNSEEN})
    assert machine.state is TrackingState.TEMPORARY_LOSS
    machine.force_discontinuity()
    # Despite a 5 s gap on the raw clock, the discontinuity reset means it must not
    # instantly abort: the gap clock restarts clean after the jump.
    machine.update(5.0, {"forward": UNSEEN})
    assert machine.state is TrackingState.TEMPORARY_LOSS
    machine.update(5.1, {"forward": SEEN})
    assert machine.state is TrackingState.TRACK
    assert machine.discontinuous is True


def test_force_discontinuity_does_not_revive_a_terminal_state():
    machine = _machine(max_prediction_gap_s=0.5)
    machine.update(0.0, {"forward": SEEN})
    machine.update(0.1, {"forward": UNSEEN})
    machine.update(1.0, {"forward": UNSEEN})
    assert machine.state is TrackingState.ABORT
    machine.force_discontinuity()
    assert machine.state is TrackingState.ABORT


@pytest.mark.parametrize("state", [TrackingState.ABORT, TrackingState.COMPLETE])
def test_is_terminal(state):
    machine = _machine()
    machine.state = state
    assert machine.is_terminal is True
