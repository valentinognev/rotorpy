"""Independent periodic setpoint publishing: expiry, fatal failure, and stats."""

from rotorpy.guidance.setpoint_stream import SetpointStream


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
