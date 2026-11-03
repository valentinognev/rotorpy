"""Wall-time staleness watchdogs for comms links (heartbeat, telemetry, image)."""

from rotorpy.guidance.watchdog import CommsWatchdogs, Watchdog


def test_watchdog_is_stale_before_the_first_touch():
    watchdog = Watchdog(timeout_s=1.0)
    assert watchdog.stale(now_s=0.0) is True
    assert watchdog.age(now_s=0.0) is None


def test_watchdog_is_fresh_right_after_a_touch():
    watchdog = Watchdog(timeout_s=1.0)
    watchdog.touch(now_s=10.0)
    assert watchdog.stale(now_s=10.2) is False
    assert abs(watchdog.age(now_s=10.2) - 0.2) < 1e-9


def test_watchdog_becomes_stale_once_the_timeout_elapses():
    watchdog = Watchdog(timeout_s=1.0)
    watchdog.touch(now_s=10.0)
    assert watchdog.stale(now_s=11.01) is True


def test_watchdog_resets_on_each_touch():
    watchdog = Watchdog(timeout_s=1.0)
    watchdog.touch(now_s=10.0)
    watchdog.touch(now_s=10.9)
    assert watchdog.stale(now_s=11.5) is False


def test_comms_watchdogs_tracks_named_links_independently():
    watchdogs = CommsWatchdogs({"heartbeat": 1.0, "telemetry": 0.5, "image": 1.0})
    watchdogs.touch("heartbeat", now_s=0.0)
    watchdogs.touch("telemetry", now_s=0.0)
    watchdogs.touch("image", now_s=0.0)
    # Only telemetry has gone stale at t=0.6.
    assert watchdogs.stale("heartbeat", now_s=0.6) is False
    assert watchdogs.stale("telemetry", now_s=0.6) is True
    assert watchdogs.stale("image", now_s=0.6) is False
    stale = watchdogs.any_stale(now_s=0.6)
    assert stale == {"telemetry": True}


def test_comms_watchdogs_reports_all_stale_before_any_touch():
    watchdogs = CommsWatchdogs({"heartbeat": 1.0, "telemetry": 0.5})
    stale = watchdogs.any_stale(now_s=0.0)
    assert stale == {"heartbeat": True, "telemetry": True}


def test_comms_watchdogs_unknown_link_raises_key_error():
    watchdogs = CommsWatchdogs({"heartbeat": 1.0})
    try:
        watchdogs.touch("unknown", now_s=0.0)
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_lost_reports_only_links_that_were_seen_and_then_went_quiet():
    # `any_stale` also reports a never-touched link (nothing to be fresh about yet); `lost`
    # is for "was alive, now silent", the condition a latching failure should key on.
    watchdogs = CommsWatchdogs({"heartbeat": 1.0, "telemetry": 0.5, "image": 1.0})
    watchdogs.touch("heartbeat", now_s=0.0)
    watchdogs.touch("telemetry", now_s=0.0)
    assert watchdogs.any_stale(now_s=0.2) == {"image": True}
    assert watchdogs.lost(now_s=0.2) == {}
    assert watchdogs.lost(now_s=0.7) == {"telemetry": True}
    assert watchdogs.lost(now_s=2.0) == {"heartbeat": True, "telemetry": True}
