"""Exactly one outcome per run, and flush/close on every termination path."""

import json

import numpy as np
import pytest

from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.run_failure import (
    EstimatorFailure,
    OffboardFailure,
    RecoverableBudget,
    RecoverableError,
    SimulatorFailure,
    StartupFailure,
    failure_reason,
    is_recoverable,
    outcome_flags,
)
from rotorpy.guidance.run_lifecycle import RunLifecycle
from rotorpy.guidance.telemetry import (
    FrameStore,
    RunMetadata,
    RunRecorder,
    TelemetryLog,
    read_telemetry,
)


class _Sink(object):
    def __init__(self, ok=True, raises=None):
        self.ok = ok
        self.raises = raises
        self.records = []

    def __call__(self, outcome, reason, fields):
        if self.raises is not None:
            raise self.raises
        self.records.append((outcome, reason, dict(fields)))
        return self.ok


def _lifecycle(sink=None, closers=(), lines=None):
    sink = _Sink() if sink is None else sink
    lines = [] if lines is None else lines
    return RunLifecycle(sink, closers=closers, announce=lines.append), sink, lines


def test_only_the_first_outcome_is_recorded():
    lifecycle, sink, _ = _lifecycle()
    assert lifecycle.finish(RunOutcome.GROUND_MISS, "ground", miss_m=3.0) is True
    assert lifecycle.finish(RunOutcome.TIMEOUT, "timeout") is False
    assert lifecycle.finish(RunOutcome.STARTUP_FAILURE, "late") is False
    assert [record[0] for record in sink.records] == [RunOutcome.GROUND_MISS]
    assert lifecycle.outcome is RunOutcome.GROUND_MISS
    assert lifecycle.outcome_reason == "ground"


def test_the_outcome_is_announced_even_when_the_sink_fails_or_raises():
    for sink in (_Sink(ok=False), _Sink(raises=OSError("disk full"))):
        lifecycle, _, lines = _lifecycle(sink)
        assert lifecycle.finish(RunOutcome.TARGET_LOST, "lost") is False
        assert lifecycle.outcome is RunOutcome.TARGET_LOST  # decided, even if not persisted
        assert not lifecycle.outcome_persisted
        assert len(lines) == 1
        payload = json.loads(lines[0].split(" ", 1)[1])
        assert payload["outcome"] == "TARGET_LOST" and payload["reason"] == "lost"
        assert lifecycle.finish(RunOutcome.TIMEOUT, "again") is False  # still exactly one
        assert len(lines) == 1


def test_missing_sink_still_yields_one_announced_outcome():
    lines = []
    lifecycle = RunLifecycle(None, announce=lines.append)
    lifecycle.finish(RunOutcome.TIMEOUT, "timeout")
    lifecycle.finish(RunOutcome.GROUND_MISS, "ground")
    assert len(lines) == 1 and lifecycle.outcome is RunOutcome.TIMEOUT


def test_a_failing_announce_never_raises():
    def broken(line):
        raise OSError("stdout closed")

    lifecycle = RunLifecycle(_Sink(), announce=broken)
    assert lifecycle.finish(RunOutcome.TIMEOUT, "timeout") is True


def test_every_closer_runs_even_when_earlier_ones_fail_and_close_is_idempotent():
    order = []

    def good(name):
        return lambda: order.append(name)

    def bad():
        order.append("bad")
        raise OSError("flush failed")

    lifecycle, _, _ = _lifecycle(closers=[("frames", good("frames")), ("telemetry", bad), ("worker", good("worker"))])
    errors = lifecycle.close()
    assert order == ["frames", "bad", "worker"]
    assert [name for name, _ in errors] == ["telemetry"]
    assert lifecycle.close() == []  # second call does nothing
    assert order == ["frames", "bad", "worker"]


def test_shutdown_without_an_outcome_writes_exactly_one_infrastructure_outcome():
    lifecycle, sink, _ = _lifecycle()
    lifecycle.ensure_outcome("shutdown_before_outcome")
    lifecycle.ensure_outcome("shutdown_before_outcome")
    assert [(r[0], r[1]) for r in sink.records] == [(RunOutcome.SIMULATOR_FAILURE, "shutdown_before_outcome")]


def test_ensure_outcome_keeps_an_existing_outcome():
    lifecycle, sink, _ = _lifecycle()
    lifecycle.finish(RunOutcome.TARGET_REACHED, "target")
    lifecycle.ensure_outcome("shutdown_before_outcome")
    assert [r[0] for r in sink.records] == [RunOutcome.TARGET_REACHED]


def test_terminate_records_then_closes_in_that_order():
    order = []
    lifecycle = RunLifecycle(
        lambda outcome, reason, fields: order.append("outcome") or True,
        closers=[("t", lambda: order.append("close"))], announce=lambda line: None,
    )
    lifecycle.terminate(RunOutcome.STARTUP_FAILURE, "estimate_timeout")
    assert order == ["outcome", "close"]


def test_fatal_errors_map_to_their_outcome_flags():
    assert outcome_flags(StartupFailure("x")) == {"startup_failed": True}
    assert outcome_flags(OffboardFailure("x")) == {"offboard_failed": True}
    assert outcome_flags(EstimatorFailure("x")) == {"estimator_failed": True}
    assert outcome_flags(SimulatorFailure("x")) == {"simulator_failed": True}
    assert outcome_flags(KeyError("boom")) == {"simulator_failed": True}
    assert failure_reason(StartupFailure("estimate_timeout")) == "estimate_timeout"
    assert failure_reason(KeyError("boom")).startswith("unexpected:KeyError")


def test_only_deliberate_recoverable_errors_are_recoverable():
    assert is_recoverable(RecoverableError("blip"))
    assert not is_recoverable(RuntimeError("blip"))
    assert not is_recoverable(StartupFailure("x"))


def test_recoverable_budget_is_bounded_consecutively_and_in_total():
    budget = RecoverableBudget(max_consecutive=3, max_total=5)
    assert budget.note(RecoverableError("a")) is None
    assert budget.note(RecoverableError("b")) is None
    budget.ok()
    assert budget.consecutive == 0
    assert budget.note(RecoverableError("c")) is None
    assert budget.note(RecoverableError("d")) is None
    assert budget.note(RecoverableError("e")) is None
    # Sixth overall error: over the total bound even though it is not consecutive.
    escalation = budget.note(RecoverableError("f"))
    assert isinstance(escalation, SimulatorFailure)
    assert "recoverable_errors" in escalation.reason

    budget = RecoverableBudget(max_consecutive=2, max_total=100)
    budget.note(RecoverableError("a"))
    budget.note(RecoverableError("b"))
    assert isinstance(budget.note(RecoverableError("c")), SimulatorFailure)


def _recorder(tmp_path):
    from rotorpy.guidance.telemetry import start_run_telemetry

    return start_run_telemetry(tmp_path, {"camera_rate_hz": 20, "timeout_s": 20.0}, environ={})


def test_telemetry_writes_exactly_one_outcome_record(tmp_path):
    recorder = _recorder(tmp_path)
    assert recorder.finish(RunOutcome.GROUND_MISS, reason="ground", miss_m=3.0)
    assert not recorder.finish(RunOutcome.TARGET_REACHED, reason="target", miss_m=0.0)
    recorder.close()
    lines = [json.loads(line) for line in (recorder.log.path).read_text().splitlines()]
    outcomes = [line for line in lines if line["kind"] == "outcome"]
    assert len(outcomes) == 1
    assert outcomes[0]["outcome"] == "GROUND_MISS"
    ignored = [line for line in lines if line.get("kind") == "outcome_ignored"]
    assert len(ignored) == 1 and ignored[0]["outcome"] == "TARGET_REACHED"
    assert read_telemetry(recorder.log.path).outcome["outcome"] == "GROUND_MISS"


def test_a_failed_outcome_write_may_be_retried_but_never_duplicated(tmp_path):
    recorder = _recorder(tmp_path)
    real_file = recorder.log._file

    class Flaky(object):
        failures = 1

        def write(self, text):
            if Flaky.failures:
                Flaky.failures -= 1
                raise OSError("disk full")
            return real_file.write(text)

        def flush(self):
            return real_file.flush()

        def close(self):
            return real_file.close()

    recorder.log._file = Flaky()
    assert not recorder.finish(RunOutcome.TIMEOUT, reason="timeout")
    assert recorder.finish(RunOutcome.TIMEOUT, reason="timeout")
    recorder.close()
    outcomes = [json.loads(l) for l in recorder.log.path.read_text().splitlines() if '"outcome"' in l]
    assert len([o for o in outcomes if o["kind"] == "outcome"]) == 1


def test_recorder_close_closes_the_handle_and_is_idempotent(tmp_path):
    recorder = _recorder(tmp_path)
    handle = recorder.log._file
    recorder.close()
    assert handle.closed
    assert not recorder.enabled
    recorder.close()  # no error
    assert not recorder.event("late")  # writes after close are refused, not raised


def test_recorder_close_saves_the_frames_held_for_a_pre_event_window(tmp_path):
    recorder = _recorder(tmp_path)
    frame = np.full((4, 4, 3), 7, dtype=np.uint8)
    recorder.frames.offer(1, "forward", frame)
    assert not recorder.frames.exists(FrameStore.name(1, "forward"))
    recorder.close(flush_frames=True)
    assert recorder.frames.exists(FrameStore.name(1, "forward"))


def test_recorder_close_survives_a_failing_handle_and_frame_store(tmp_path):
    recorder = _recorder(tmp_path)

    class Broken(object):
        closed = False

        def flush(self):
            raise OSError("flush failed")

        def close(self):
            raise OSError("close failed")

    recorder.log._file = Broken()

    def failing_mark():
        raise OSError("frame disk full")

    recorder.frames.mark_event = failing_mark
    recorder.close(flush_frames=True)  # must not raise
    assert not recorder.enabled
    assert recorder.log.last_error is not None


def test_a_disabled_recorder_closes_quietly(tmp_path):
    bad_root = tmp_path / "file"
    bad_root.write_text("not a directory")
    from rotorpy.guidance.telemetry import start_run_telemetry

    recorder = start_run_telemetry(bad_root, {"camera_rate_hz": 20}, environ={})
    assert not recorder.enabled
    recorder.close(flush_frames=True)
    assert not recorder.finish(RunOutcome.TIMEOUT, reason="timeout")


def test_service_unavailable_is_recoverable_and_still_a_runtime_error():
    from rotorpy.guidance.run_failure import ServiceUnavailable

    exc = ServiceUnavailable("get_model_state failed")
    assert is_recoverable(exc) and isinstance(exc, RuntimeError)
