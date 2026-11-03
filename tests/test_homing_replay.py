"""Record a synthetic flight, then reproduce its decisions without Gazebo."""

import os

import numpy as np

from rotorpy.guidance.homing_config import guidance_from_config, load_homing_config
from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.replay import replay_run
from rotorpy.guidance.telemetry import (
    FrameStore,
    RunMetadata,
    RunRecorder,
    TelemetryLog,
    frame_to_float,
    read_telemetry,
)

_SIZE = 128
_ATTITUDE = np.array([0.0, 0.0, 0.0, 1.0])
_VELOCITY = np.array([4.0, 1.0, 0.0])


def _config():
    config = load_homing_config(os.path.join(
        os.path.dirname(__file__), "..", "examples", "camera_homing.jsonc"
    ))
    config["image_width_px"] = _SIZE
    config["image_height_px"] = _SIZE
    return config


def _frame(center=None, side=10):
    frame = np.full((_SIZE, _SIZE, 3), 30, dtype=np.uint8)
    if center is not None:
        column, row = center
        frame[row:row + side, column:column + side] = (230, 20, 20)
    return frame


def _scenario():
    """(forward, upward, updated) per tick: blank, forward lock, handoff, loss, stale ticks."""
    ticks = []
    for _ in range(3):
        ticks.append((_frame(), _frame(), True))
    for i in range(6):
        ticks.append((_frame((40 + 3 * i, 50 + 2 * i)), _frame(), True))
        ticks.append((None, None, False))
    for i in range(3):
        ticks.append((_frame((30, 40)), _frame((60 + 4 * i, 55 - 3 * i)), True))
    ticks.append((_frame(), _frame(), True))
    return ticks


def _record(tmp_path, frame_store_kwargs=None):
    config = _config()
    log = TelemetryLog.create(tmp_path, RunMetadata(
        config=config, firmware_id="test", px4_params={}, rendering_mode="none", sim_timing={},
    ))
    frames = FrameStore(log.run_dir, **(frame_store_kwargs or {"save_all": True}))
    recorder = RunRecorder(log, frames)
    guidance = guidance_from_config(config)
    last_forward = last_upward = None
    for seq, (forward, upward, updated) in enumerate(_scenario()):
        t = 0.05 * seq
        if updated:
            last_forward, last_upward = forward, upward
            command = guidance.update(
                t, 30.0, _ATTITUDE, _VELOCITY, frame_to_float(forward), frame_to_float(upward)
            )
        assert recorder.step(
            seq=seq, t_s=t, height_m=30.0, attitude_quat=_ATTITUDE, velocity_ned=_VELOCITY,
            forward=last_forward if updated else None, upward=last_upward if updated else None,
            updated=updated, command=command,
        )
    log.finish(RunOutcome.TARGET_LOST, reason="synthetic")
    log.close()
    return log


def test_replay_reproduces_recorded_decisions_exactly(tmp_path):
    log = _record(tmp_path)
    result = replay_run(log.path)
    run = read_telemetry(log.path)
    assert result.gap_seq is None
    assert result.matches
    assert result.replayed == sum(1 for tick in run.ticks if tick.guidance_update is not None)
    assert result.replayed == 13
    assert result.skipped == 6
    assert [d["seq"] for d in result.decisions] == [
        tick.seq for tick in run.ticks if tick.guidance_update is not None
    ]
    for decision, tick in zip(result.decisions, [t for t in run.ticks if t.guidance_update]):
        assert decision["decision"] == tick.decision
    assert any(tick.decision["tracking"] for tick in run.ticks if tick.decision)


def test_replay_flags_a_different_guidance_as_a_mismatch(tmp_path):
    log = _record(tmp_path)

    def other_guidance(config):
        changed = dict(config)
        changed["nav_constant"] = float(config["nav_constant"]) + 3.0
        return guidance_from_config(changed)

    result = replay_run(log.path, guidance_factory=other_guidance)
    assert not result.matches
    assert result.mismatches


def test_replay_stops_at_the_first_missing_frame(tmp_path):
    log = _record(tmp_path, {"pre": 4, "post": 1})
    result = replay_run(log.path)
    assert result.gap_seq == 7
    assert result.replayed == 5
    assert result.matches


def test_recording_through_a_broken_log_never_raises(tmp_path):
    config = _config()
    log = TelemetryLog.start_tolerant(tmp_path / "x" / "y", RunMetadata(
        config=config, firmware_id="t", px4_params={}, rendering_mode="n", sim_timing={},
    ))
    recorder = RunRecorder(log, FrameStore(tmp_path / "x" / "y", save_all=True))
    assert recorder.step(
        seq=0, t_s=0.0, height_m=1.0, attitude_quat=_ATTITUDE, velocity_ned=_VELOCITY,
        forward=_frame(), upward=None, updated=True, command=None,
    ) is False
