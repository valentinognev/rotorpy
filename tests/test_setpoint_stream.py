"""Independent periodic setpoint publishing: expiry, fatal failure, and stats."""

import threading

from rotorpy.guidance.setpoint_stream import PROPOSED_FAILURE_COMMAND, SetpointStream


def test_tick_before_any_command_is_set_sends_nothing_by_default():
    sent = []
    stream = SetpointStream(publish=sent.append, expiry_s=1.0)
    result = stream.tick(now_s=0.0)
    assert result is None
    assert sent == []
    assert stream.stats.expired_count == 0  # no command was ever set, so nothing "expired"


def test_tick_publishes_the_latest_fresh_command():
    sent = []
    stream = SetpointStream(publish=sent.append, expiry_s=1.0)
    stream.set_command("v1", observation_wall_s=0.0)
    result = stream.tick(now_s=0.1)
    assert result == "v1"
    assert sent == ["v1"]


def test_command_expires_after_the_expiry_window_and_sends_the_failure_command():
    sent = []
    stream = SetpointStream(publish=sent.append, expiry_s=0.5, failure_command="hold")
    stream.set_command("v1", observation_wall_s=0.0)
    stream.tick(now_s=0.1)
    result = stream.tick(now_s=0.6)
    assert result == "hold"
    assert sent[-1] == "hold"
    assert stream.stats.expired_count == 1


def test_default_failure_response_sends_nothing():
    sent = []
    stream = SetpointStream(publish=sent.append, expiry_s=0.5)
    stream.set_command("v1", observation_wall_s=0.0)
    result = stream.tick(now_s=0.6)
    assert result is None
    assert sent == []


def test_a_fresh_command_after_expiry_resumes_normal_publishing():
    sent = []
    stream = SetpointStream(publish=sent.append, expiry_s=0.5, failure_command="hold")
    stream.set_command("v1", observation_wall_s=0.0)
    stream.tick(now_s=0.6)
    stream.set_command("v2", observation_wall_s=0.6)
    result = stream.tick(now_s=0.65)
    assert result == "v2"


def test_mark_fatal_stops_the_normal_command_even_with_a_fresh_observation():
    sent = []
    stream = SetpointStream(publish=sent.append, expiry_s=1.0, failure_command="hold")
    stream.set_command("v1", observation_wall_s=0.0)
    stream.mark_fatal()
    result = stream.tick(now_s=0.01)
    assert result == "hold"
    assert stream.stats.failed is True


def test_mark_fatal_records_the_first_reason_in_stats_and_snapshot():
    # The latch has no un-set path, so the reason it tripped must be visible afterwards.
    stream = SetpointStream(publish=lambda c: None, expiry_s=10.0)
    assert stream.stats_snapshot()["fatal_reason"] is None
    stream.mark_fatal("watchdog_lost:telemetry")
    stream.mark_fatal("step_exception: boom")
    assert stream.stats.fatal_reason == "watchdog_lost:telemetry"
    assert stream.stats_snapshot()["fatal_reason"] == "watchdog_lost:telemetry"


def test_stats_track_publish_intervals():

    sent = []
    stream = SetpointStream(publish=sent.append, expiry_s=10.0)
    stream.set_command("v1", observation_wall_s=0.0)
    stream.tick(now_s=0.0)
    stream.tick(now_s=0.1)
    stream.tick(now_s=0.25)
    assert stream.stats.publish_count == 3
    assert stream.stats.last_interval_s is not None
    assert abs(stream.stats.last_interval_s - 0.15) < 1e-9
    assert abs(stream.stats.max_interval_s - 0.15) < 1e-9


def test_stats_track_end_to_end_observation_age():
    sent = []
    stream = SetpointStream(publish=sent.append, expiry_s=10.0)
    stream.set_command("v1", observation_wall_s=0.0)
    stream.tick(now_s=0.3)
    assert abs(stream.stats.last_observation_age_s - 0.3) < 1e-9
    assert abs(stream.stats.max_observation_age_s - 0.3) < 1e-9
    stream.set_command("v2", observation_wall_s=0.3)
    stream.tick(now_s=0.32)
    assert abs(stream.stats.last_observation_age_s - 0.02) < 1e-9
    assert abs(stream.stats.max_observation_age_s - 0.3) < 1e-9  # max persists


def test_fake_clock_is_used_when_now_is_omitted():
    times = iter([100.0, 100.2, 100.5])
    stream = SetpointStream(publish=lambda c: None, expiry_s=10.0, now=lambda: next(times))
    stream.set_command("v1", observation_wall_s=100.0)
    stream.tick()
    stream.tick()
    assert stream.stats.publish_count == 2


def test_default_failure_response_is_the_documented_proposed_constant():
    # The "send nothing on failure" default is a named, documented decision, not an
    # unexamined `None` default argument.
    assert PROPOSED_FAILURE_COMMAND is None
    stream = SetpointStream(publish=lambda c: None, expiry_s=10.0)
    assert stream._failure_command is PROPOSED_FAILURE_COMMAND


def test_stats_snapshot_is_a_plain_dict_usable_for_telemetry():
    stream = SetpointStream(publish=lambda c: None, expiry_s=10.0)
    stream.set_command("v1", observation_wall_s=0.0)
    stream.tick(now_s=0.0)
    snapshot = stream.stats_snapshot()
    assert isinstance(snapshot, dict)
    assert snapshot["publish_count"] == 1
    assert snapshot["failed"] is False


def test_stats_do_not_count_ticks_that_published_nothing():
    # Before this fix, `publish_count`/interval stats advanced on every `tick()`, including
    # ticks that sent nothing (no command ever set, and no failure command configured) --
    # over-reporting the publisher as healthy.
    stream = SetpointStream(publish=lambda c: None, expiry_s=10.0)
    stream.tick(now_s=0.0)
    stream.tick(now_s=1.0)
    assert stream.stats.publish_count == 0
    assert stream.stats.last_interval_s is None
    assert stream.stats.max_interval_s is None
    stream.set_command("v1", observation_wall_s=1.0)
    stream.tick(now_s=1.0)  # first real publish
    assert stream.stats.publish_count == 1
    assert stream.stats.last_interval_s is None  # only one real publish so far: no interval yet
    stream.tick(now_s=1.2)  # second real publish
    assert stream.stats.publish_count == 2
    assert abs(stream.stats.last_interval_s - 0.2) < 1e-9


def test_set_command_and_tick_pair_command_and_time_atomically_under_concurrency():
    # `set_command` writes the command and its wall time as one atomic unit; a concurrent
    # `peek` (what the 50 Hz publisher thread's `tick()` reads) must never observe one
    # without the other. The writer deliberately widens the real race window *inside the
    # real lock* (not a reimplementation) so a reader hammering `peek()` for the writer's
    # entire lifetime gets many genuine chances to observe a half-updated pair if the two
    # fields were not actually serialised.
    import time as time_module

    class _SlowSetCommand(SetpointStream):
        def set_command(self, command, observation_wall_s):
            with self._lock:
                self._command = command
                time_module.sleep(0.001)
                self._command_wall_s = float(observation_wall_s)

    stream = _SlowSetCommand(publish=lambda c: None, expiry_s=1e9)
    mismatches = []
    done = threading.Event()

    def writer():
        for i in range(30):
            stream.set_command(i, observation_wall_s=float(i))
        done.set()

    def reader():
        while not done.is_set():
            command, stamp_wall_s = stream.peek()
            if command is not None and stamp_wall_s is not None and command != stamp_wall_s:
                mismatches.append((command, stamp_wall_s))

    writer_thread = threading.Thread(target=writer)
    reader_thread = threading.Thread(target=reader)
    reader_thread.start()
    writer_thread.start()
    writer_thread.join()
    reader_thread.join()
    assert mismatches == []
