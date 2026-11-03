"""Clock model, attitude history interpolation and PX4-telemetry-driven vehicle state."""

import sys
import threading
import time

import numpy as np
import pytest

from rotorpy.guidance.origin_alignment import EstimatorOriginAlignment
from rotorpy.guidance.vehicle_state import AttitudeHistory, ClockModel, VehicleState


# --- ClockModel -------------------------------------------------------------------


def test_clock_model_reports_no_discontinuity_on_first_sample():
    clock = ClockModel(backward_jump_s=0.05, pause_wall_s=0.5)
    assert clock.update(sim_s=0.0, wall_s=100.0) is None


def test_clock_model_reports_no_discontinuity_for_normal_advance():
    clock = ClockModel(backward_jump_s=0.05, pause_wall_s=0.5)
    clock.update(sim_s=1.0, wall_s=100.0)
    assert clock.update(sim_s=1.05, wall_s=100.05) is None


def test_clock_model_detects_a_backward_jump():
    clock = ClockModel(backward_jump_s=0.05, pause_wall_s=0.5)
    clock.update(sim_s=10.0, wall_s=100.0)
    assert clock.update(sim_s=9.5, wall_s=100.05) == "backward_jump"


def test_clock_model_detects_a_reset_to_near_zero():
    clock = ClockModel(backward_jump_s=0.05, pause_wall_s=0.5)
    clock.update(sim_s=42.0, wall_s=100.0)
    assert clock.update(sim_s=0.0, wall_s=100.05) == "reset"


def test_clock_model_detects_a_pause_when_wall_time_advances_without_sim_time():
    clock = ClockModel(backward_jump_s=0.05, pause_wall_s=0.5)
    clock.update(sim_s=10.0, wall_s=100.0)
    assert clock.update(sim_s=10.0, wall_s=100.6) == "paused"


def test_clock_model_small_jitter_is_not_a_discontinuity():
    clock = ClockModel(backward_jump_s=0.05, pause_wall_s=0.5)
    clock.update(sim_s=10.0, wall_s=100.0)
    assert clock.update(sim_s=9.99, wall_s=100.01) is None


# --- AttitudeHistory ---------------------------------------------------------------


def _yaw_quat(yaw_rad):
    return np.array([0.0, 0.0, np.sin(yaw_rad / 2.0), np.cos(yaw_rad / 2.0)])


def test_attitude_history_interpolates_between_two_samples():
    history = AttitudeHistory(maxlen=10)
    history.record(0.0, _yaw_quat(0.0))
    history.record(1.0, _yaw_quat(np.deg2rad(90.0)))
    quat, in_range = history.interpolate(0.5)
    assert in_range is True
    yaw = 2.0 * np.arctan2(quat[2], quat[3])
    assert np.isclose(yaw, np.deg2rad(45.0), atol=1e-6)


def test_attitude_history_flags_out_of_range_but_still_returns_the_nearest_sample():
    history = AttitudeHistory(maxlen=10)
    history.record(0.0, _yaw_quat(0.0))
    history.record(1.0, _yaw_quat(np.deg2rad(90.0)))
    quat, in_range = history.interpolate(5.0)
    assert in_range is False
    assert quat is not None


def test_attitude_history_empty_returns_none_and_out_of_range():
    history = AttitudeHistory(maxlen=10)
    quat, in_range = history.interpolate(0.0)
    assert quat is None
    assert in_range is False


def test_attitude_history_is_bounded_and_drops_the_oldest_sample():
    history = AttitudeHistory(maxlen=2)
    history.record(0.0, _yaw_quat(0.0))
    history.record(1.0, _yaw_quat(np.deg2rad(45.0)))
    history.record(2.0, _yaw_quat(np.deg2rad(90.0)))
    assert len(history) == 2
    _, in_range = history.interpolate(0.0)
    # The t=0.0 sample was dropped; 0.0 is now before the buffered window.
    assert in_range is False


def test_attitude_history_clear_empties_it():
    history = AttitudeHistory(maxlen=10)
    history.record(0.0, _yaw_quat(0.0))
    history.clear()
    assert len(history) == 0
    quat, in_range = history.interpolate(0.0)
    assert quat is None
    assert in_range is False


def test_attitude_history_record_does_not_crash_on_a_duplicate_timestamp():
    # Several PX4 attitude messages drained in the same quantised sim tick share one
    # `rospy.get_time()` value; `interpolate` requires strictly increasing times, so a
    # naive append would crash the very next `sample_at` call with a fresh image.
    history = AttitudeHistory(maxlen=10)
    history.record(1.0, _yaw_quat(0.0))
    history.record(1.0, _yaw_quat(np.deg2rad(10.0)))
    assert len(history) == 1
    quat, in_range = history.interpolate(1.0)
    assert in_range is True
    # The newest attitude for that instant wins, not the first one recorded.
    yaw = 2.0 * np.arctan2(quat[2], quat[3])
    assert np.isclose(yaw, np.deg2rad(10.0), atol=1e-6)


def test_attitude_history_record_does_not_crash_on_an_out_of_order_timestamp():
    history = AttitudeHistory(maxlen=10)
    history.record(2.0, _yaw_quat(0.0))
    # Not a clock-model discontinuity (too small a regression to be flagged), just jitter
    # within one sim tick: must not append a non-increasing timestamp.
    history.record(1.999, _yaw_quat(np.deg2rad(20.0)))
    assert len(history) == 1
    quat, in_range = history.interpolate(2.0)
    assert in_range is True


# --- VehicleState -------------------------------------------------------------------


def _alignment(yaw_deg=0.0):
    return EstimatorOriginAlignment(
        origin_world_ned=[0.0, 0.0, -30.0], yaw_offset_rad=np.deg2rad(yaw_deg), ground_down_world_m=0.0
    )


def test_vehicle_state_interpolates_attitude_at_capture_time():
    state = VehicleState(_alignment())
    state.ingest_attitude(sim_s=0.0, wall_s=0.0, quat_estimator_xyzw=_yaw_quat(0.0))
    state.ingest_attitude(sim_s=1.0, wall_s=1.0, quat_estimator_xyzw=_yaw_quat(np.deg2rad(90.0)))
    sample = state.sample_at(capture_sim_s=0.5, now_wall_s=1.0)
    assert sample.attitude_in_range is True
    yaw = 2.0 * np.arctan2(sample.attitude_quat_world[2], sample.attitude_quat_world[3])
    assert np.isclose(yaw, np.deg2rad(45.0), atol=1e-6)


def test_vehicle_state_applies_yaw_offset_to_attitude():
    state = VehicleState(_alignment(yaw_deg=90.0))
    state.ingest_attitude(sim_s=0.0, wall_s=0.0, quat_estimator_xyzw=_yaw_quat(0.0))
    sample = state.sample_at(capture_sim_s=0.0, now_wall_s=0.0)
    yaw = 2.0 * np.arctan2(sample.attitude_quat_world[2], sample.attitude_quat_world[3])
    assert np.isclose(yaw, np.deg2rad(90.0), atol=1e-6)


def test_vehicle_state_converts_local_position_into_the_world_frame():
    state = VehicleState(_alignment())
    state.ingest_local_position(
        wall_s=2.0, position_estimator_ned=[10.0, 5.0, -20.0], velocity_estimator_ned=[1.0, 0.0, 0.0]
    )
    sample = state.sample_at(capture_sim_s=None, now_wall_s=2.0)
    assert np.allclose(sample.position_ned, [10.0, 5.0, -50.0])
    assert np.isclose(sample.height_above_ground_m, 50.0)


def test_vehicle_state_reports_ages_from_wall_clock():
    state = VehicleState(_alignment())
    state.ingest_attitude(sim_s=0.0, wall_s=10.0, quat_estimator_xyzw=_yaw_quat(0.0))
    state.ingest_local_position(wall_s=10.0, position_estimator_ned=[0.0, 0.0, 0.0], velocity_estimator_ned=[0.0, 0.0, 0.0])
    sample = state.sample_at(capture_sim_s=0.0, now_wall_s=10.4)
    assert np.isclose(sample.attitude_age_s, 0.4)
    assert np.isclose(sample.position_age_s, 0.4)


def test_vehicle_state_sample_before_any_data_has_none_fields():
    state = VehicleState(_alignment())
    sample = state.sample_at(capture_sim_s=0.0, now_wall_s=0.0)
    assert sample.attitude_quat_world is None
    assert sample.attitude_in_range is False
    assert sample.position_ned is None
    assert sample.height_above_ground_m is None
    assert sample.attitude_age_s is None
    assert sample.position_age_s is None


def test_vehicle_state_sample_without_a_capture_time_uses_the_latest_attitude():
    state = VehicleState(_alignment())
    state.ingest_attitude(sim_s=0.0, wall_s=0.0, quat_estimator_xyzw=_yaw_quat(0.0))
    state.ingest_attitude(sim_s=1.0, wall_s=1.0, quat_estimator_xyzw=_yaw_quat(np.deg2rad(90.0)))
    # No image this tick (`capture_sim_s=None`): still report the most recent attitude,
    # not "no attitude", so startup heading and the gravity/arming state machine keep working.
    sample = state.sample_at(capture_sim_s=None, now_wall_s=1.0)
    assert sample.attitude_in_range is True
    yaw = 2.0 * np.arctan2(sample.attitude_quat_world[2], sample.attitude_quat_world[3])
    assert np.isclose(yaw, np.deg2rad(90.0), atol=1e-6)


def test_vehicle_state_clock_discontinuity_clears_attitude_history():
    state = VehicleState(_alignment(), clock_backward_jump_s=0.05, clock_pause_wall_s=0.5)
    state.ingest_attitude(sim_s=10.0, wall_s=10.0, quat_estimator_xyzw=_yaw_quat(0.0))
    assert state.last_discontinuity is None
    # A backward jump beyond the threshold clears the old attitude history.
    state.ingest_attitude(sim_s=1.0, wall_s=10.05, quat_estimator_xyzw=_yaw_quat(np.deg2rad(10.0)))
    assert state.last_discontinuity == "backward_jump"
    sample = state.sample_at(capture_sim_s=10.0, now_wall_s=10.05)
    # The sim_s=10.0 sample from before the discontinuity is gone.
    assert sample.attitude_in_range is False
    assert state.take_discontinuity() == "backward_jump"


def test_discontinuity_is_sticky_until_taken():
    # Several MAVLink messages are drained (several `ingest_attitude` calls) before the
    # tracker's single `sample_at` call for that tick: a discontinuity flagged by an
    # earlier ingest in the same drain must not be overwritten back to `None` by a later
    # ingest in the same drain that detects no further discontinuity.
    state = VehicleState(_alignment(), clock_backward_jump_s=0.05, clock_pause_wall_s=0.5)
    state.ingest_attitude(sim_s=10.0, wall_s=10.0, quat_estimator_xyzw=_yaw_quat(0.0))
    state.ingest_attitude(sim_s=1.0, wall_s=10.05, quat_estimator_xyzw=_yaw_quat(np.deg2rad(10.0)))
    assert state.last_discontinuity == "backward_jump"
    # A second, unrelated ingest in the same drain reports no discontinuity of its own...
    state.ingest_attitude(sim_s=1.01, wall_s=10.06, quat_estimator_xyzw=_yaw_quat(np.deg2rad(11.0)))
    assert state.last_discontinuity is None
    # ...but the pending discontinuity from the first ingest must still reach its consumer.
    assert state.take_discontinuity() == "backward_jump"
    # Consumed: the next `take_discontinuity` call reports no (new) discontinuity.
    assert state.take_discontinuity() is None


# --- discontinuity consumption + thread safety (fix round 2) --------------------------


def _discontinuous_state():
    state = VehicleState(_alignment(), clock_backward_jump_s=0.05, clock_pause_wall_s=0.5)
    state.ingest_attitude(sim_s=10.0, wall_s=10.0, quat_estimator_xyzw=_yaw_quat(0.0))
    state.ingest_attitude(sim_s=1.0, wall_s=10.05, quat_estimator_xyzw=_yaw_quat(0.0))
    return state


def test_sample_at_is_side_effect_free_for_the_discontinuity_flag():
    # The publisher thread calls `sample_at` too; it must never consume the flag the
    # step thread has yet to observe.
    state = _discontinuous_state()
    for _ in range(3):
        state.sample_at(capture_sim_s=None, now_wall_s=10.06)
    assert state.take_discontinuity() == "backward_jump"
    assert state.take_discontinuity() is None


def test_concurrent_sample_at_cannot_swallow_the_discontinuity():
    state = _discontinuous_state()
    stop = threading.Event()
    started = threading.Event()

    def publisher():
        started.set()
        while not stop.is_set():
            state.sample_at(capture_sim_s=None, now_wall_s=10.06)

    thread = threading.Thread(target=publisher)
    thread.start()
    started.wait()
    time.sleep(0.05)
    try:
        assert state.take_discontinuity() == "backward_jump"
    finally:
        stop.set()
        thread.join()


def test_vehicle_state_reads_are_consistent_under_concurrent_ingest():
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    state = VehicleState(_alignment(), clock_backward_jump_s=0.05, clock_pause_wall_s=0.5)
    errors = []
    done = threading.Event()

    def writer():
        try:
            for k in range(1500):
                # A periodic backward jump clears the history while the reader interpolates.
                state.ingest_attitude(sim_s=float(k % 40) * 0.01, wall_s=k * 0.001,
                                      quat_estimator_xyzw=_yaw_quat(0.1))
                state.ingest_local_position(
                    wall_s=k * 0.001, position_estimator_ned=[k, k, k], velocity_estimator_ned=[k, k, k]
                )
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)
        finally:
            done.set()

    def reader(capture):
        try:
            while not done.is_set():
                sample = state.sample_at(capture_sim_s=capture, now_wall_s=5.0)
                if sample.position_ned is not None:
                    # Position and velocity are written together: a torn read mixes ticks.
                    assert sample.position_ned[0] + 0.0 == sample.velocity_ned[0] + 0.0
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer)] + [threading.Thread(target=reader, args=(c,)) for c in (0.05, None)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    finally:
        sys.setswitchinterval(old_interval)
    assert errors == []
