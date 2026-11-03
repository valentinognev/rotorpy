"""Per-tick telemetry, run identity, frame saving and failure tolerance."""

import json
import os

import numpy as np
import pytest

from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.telemetry import (
    SCHEMA_VERSION,
    EventDetector,
    FrameStore,
    RunMetadata,
    TelemetryLog,
    TickRecord,
    frame_to_float,
    new_run_id,
    read_telemetry,
)


def _metadata():
    return RunMetadata(
        config={"nav_constant": 3.0, "start_ned": np.array([0.0, 0.0, -30.0])},
        firmware_id="PX4 v1.13.3 abc123",
        px4_params={"MPC_XY_VEL_P_ACC": 1.8},
        rendering_mode="gpu",
        sim_timing={"camera_rate_hz": 20.0, "real_time_factor": 1.0},
    )


def _tick(seq=0):
    return TickRecord(
        seq=seq,
        t_s=0.05 * seq,
        image_capture_s={"forward": 0.01, "upward": None},
        image_receipt_s={"forward": 0.06, "upward": None},
        processing_start_s=0.07,
        processing_end_s=0.08,
        attitude_quat=np.array([0.0, 0.0, 0.0, 1.0]),
        position_ned=np.array([1.0, 2.0, -30.0]),
        velocity_ned=np.array([3.0, 0.0, 0.0]),
        attitude_age_s=0.02,
        position_age_s=0.03,
        velocity_age_s=0.03,
        candidates=[{"camera": "forward", "center": [10.0, 12.0], "side": 6, "pixels": 30}],
        selected={"camera": "forward", "center": [10.0, 12.0], "side": 6},
        confidence=0.9,
        camera="forward",
        target_motion={"estimate": [0.0, 0.0], "sigma": [1.0, 1.0]},
        requested_setpoint=[5.0, 0.0, 0.0],
        limited_setpoint=[4.0, 0.0, 0.0],
        transmitted_setpoint=[4.0, 0.0, 0.0],
        limiting_reasons=["speed_envelope"],
        measured_velocity_ned=[3.0, 0.0, 0.0],
        state="TRACKING",
        px4_mode="OFFBOARD",
        failure_reason=None,
        guidance_update={"t": 0.0, "forward_frame": "frames/000000_forward.npy"},
        decision={"locked": True},
    )


def test_run_ids_are_unique_and_never_overwrite(tmp_path):
    ids = iter(["same", "same", "other"])
    first = TelemetryLog.create(tmp_path, _metadata(), id_factory=lambda: next(ids))
    first.write_event("marker")
    first.close()
    before = first.path.read_bytes()

    second = TelemetryLog.create(tmp_path, _metadata(), id_factory=lambda: next(ids))
    second.close()

    assert first.run_id == "same"
    assert second.run_id == "other"
    assert first.path.read_bytes() == before
    assert first.path != second.path


def test_default_run_ids_differ_within_one_second(tmp_path):
    assert len({new_run_id() for _ in range(50)}) == 50
    a = TelemetryLog.create(tmp_path, _metadata())
    b = TelemetryLog.create(tmp_path, _metadata())
    assert a.run_id != b.run_id
    assert a.path.exists() and b.path.exists()


def test_create_gives_up_instead_of_overwriting(tmp_path):
    TelemetryLog.create(tmp_path, _metadata(), id_factory=lambda: "fixed").close()
    with pytest.raises(FileExistsError):
        TelemetryLog.create(tmp_path, _metadata(), id_factory=lambda: "fixed", attempts=3)


def test_schema_round_trip(tmp_path):
    log = TelemetryLog.create(tmp_path, _metadata())
    tick = _tick(3)
    assert log.write_tick(tick)
    assert log.write_event("handoff", seq=3, detail="forward->upward")
    assert log.finish(RunOutcome.GROUND_MISS, reason="ground contact", miss_m=12.5)
    log.close()

    run = read_telemetry(log.path)
    assert run.metadata["schema"] == SCHEMA_VERSION
    assert run.metadata["run_id"] == log.run_id
    assert run.metadata["config"]["start_ned"] == [0.0, 0.0, -30.0]
    assert run.metadata["firmware_id"] == "PX4 v1.13.3 abc123"
    assert run.metadata["px4_params"] == {"MPC_XY_VEL_P_ACC": 1.8}
    assert run.metadata["rendering_mode"] == "gpu"
    assert run.metadata["sim_timing"]["camera_rate_hz"] == 20.0
    assert len(run.ticks) == 1
    assert run.ticks[0] == TickRecord.from_dict(json.loads(json.dumps(tick.to_dict())))
    assert run.ticks[0].velocity_ned == [3.0, 0.0, 0.0]
    assert run.ticks[0].limiting_reasons == ["speed_envelope"]
    assert run.events[0]["kind"] == "handoff"
    assert run.outcome["outcome"] == "GROUND_MISS"
    assert run.outcome["miss_m"] == 12.5
    assert run.truncated_lines == 0


def test_setpoint_stats_round_trip(tmp_path):
    # Spec: setpoint publish stats were never surfaced through telemetry, making a dead
    # setpoint stream invisible except by inferring it from a lack of commanded motion.
    log = TelemetryLog.create(tmp_path, _metadata())
    tick = _tick(0)
    tick.setpoint_stats = {
        "publish_count": 42, "last_interval_s": 0.02, "mean_interval_s": 0.02,
        "max_interval_s": 0.03, "last_observation_age_s": 0.01, "max_observation_age_s": 0.4,
        "expired_count": 1, "failed": False,
    }
    assert log.write_tick(tick)
    log.close()
    run = read_telemetry(log.path)
    assert run.ticks[0].setpoint_stats["publish_count"] == 42
    assert run.ticks[0].setpoint_stats["expired_count"] == 1


def test_tick_schema_covers_the_plan_fields():
    fields = set(TickRecord.__dataclass_fields__)
    assert {
        "image_capture_s", "image_receipt_s", "processing_start_s", "processing_end_s",
        "attitude_quat", "position_ned", "velocity_ned",
        "attitude_age_s", "position_age_s", "velocity_age_s",
        "candidates", "selected", "confidence", "camera", "target_motion",
        "requested_setpoint", "limited_setpoint", "transmitted_setpoint",
        "limiting_reasons", "measured_velocity_ned",
        "state", "px4_mode", "failure_reason", "setpoint_stats",
    } <= fields


def test_reader_ignores_unknown_fields_and_survives_a_truncated_tail(tmp_path):
    log = TelemetryLog.create(tmp_path, _metadata())
    log.write_tick(_tick(0))
    log.close()
    with open(log.path, "a") as handle:
        handle.write('{"schema": 1, "kind": "tick", "seq": 1, "future_field": 5}\n')
        handle.write('{"schema": 1, "kind": "tick", "seq": 2, "t_s": 0.')
    run = read_telemetry(log.path)
    assert [tick.seq for tick in run.ticks] == [0, 1]
    assert run.truncated_lines == 1


def test_recording_failures_never_raise(tmp_path):
    blocker = tmp_path / "not_a_directory"
    blocker.write_text("x")
    disabled = TelemetryLog.start_tolerant(blocker, _metadata())
    assert not disabled.enabled
    assert disabled.write_tick(_tick()) is False
    assert disabled.write_event("x") is False
    assert disabled.finish(RunOutcome.TIMEOUT) is False
    disabled.close()

    log = TelemetryLog.start_tolerant(tmp_path, _metadata())
    assert log.enabled
    log._file.close()
    assert log.write_tick(_tick()) is False
    assert log.last_error is not None
    for _ in range(10):
        log.write_tick(_tick())
    assert not log.enabled


def test_unserializable_values_do_not_raise(tmp_path):
    log = TelemetryLog.create(tmp_path, _metadata())
    assert log.write_event("odd", payload=object()) is True
    log.close()
    assert read_telemetry(log.path).events[0]["kind"] == "odd"


def test_frames_round_trip_losslessly(tmp_path):
    store = FrameStore(tmp_path, save_all=True)
    frame = np.random.RandomState(0).randint(0, 256, size=(7, 9, 3), dtype=np.uint8)
    name = store.name(4, "forward")
    assert store.offer(4, "forward", frame) == name
    loaded = store.load(name)
    assert loaded.dtype == np.uint8
    assert np.array_equal(loaded, frame)
    assert frame_to_float(loaded).dtype == np.float32
    assert np.array_equal(frame_to_float(loaded), frame.astype(np.float32) / 255.0)


def test_frames_around_events_are_saved_retroactively(tmp_path):
    store = FrameStore(tmp_path, pre=2, post=1)
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    for seq in range(6):
        store.offer(seq, "forward", frame)
    assert not any(store.exists(store.name(seq, "forward")) for seq in range(6))
    store.mark_event()
    assert [store.exists(store.name(seq, "forward")) for seq in range(6)] == [
        False, False, False, False, True, True,
    ]
    store.offer(6, "forward", frame)
    store.offer(7, "forward", frame)
    assert store.exists(store.name(6, "forward"))
    assert not store.exists(store.name(7, "forward"))


def test_frame_saving_failure_is_swallowed(tmp_path):
    store = FrameStore(tmp_path / "missing" / "deeper", save_all=True)
    assert store.offer(0, "forward", np.zeros((2, 2, 3), dtype=np.uint8)) is None
    assert store.last_error is not None


def test_event_detector_names_acquisition_handoff_and_loss():
    detector = EventDetector()
    assert detector.update(False, None) is None
    assert detector.update(True, "forward") == "acquisition"
    assert detector.update(True, "forward") is None
    assert detector.update(True, "upward") == "handoff"
    assert detector.update(False, None) == "loss"
    assert detector.update(False, None) is None
    assert detector.update(True, "forward") == "reacquisition"


def test_run_directory_files_stay_under_the_run(tmp_path):
    log = TelemetryLog.create(tmp_path, _metadata())
    log.close()
    assert os.path.dirname(log.path) == str(log.run_dir)
    assert log.run_dir.parent == tmp_path
