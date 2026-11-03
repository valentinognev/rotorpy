"""Proposed comms-watchdog timeouts live in homing_config, named and documented."""

from rotorpy.guidance.homing_config import (
    PROPOSED_HEARTBEAT_TIMEOUT_S,
    PROPOSED_IMAGE_TIMEOUT_S,
    PROPOSED_TELEMETRY_TIMEOUT_S,
    watchdog_timeouts,
)
from rotorpy.guidance.watchdog import CommsWatchdogs


def test_watchdog_timeouts_are_positive_and_named():
    timeouts = watchdog_timeouts()
    assert timeouts == {
        "heartbeat": PROPOSED_HEARTBEAT_TIMEOUT_S,
        "telemetry": PROPOSED_TELEMETRY_TIMEOUT_S,
        "image": PROPOSED_IMAGE_TIMEOUT_S,
    }
    assert all(value > 0.0 for value in timeouts.values())


def test_watchdog_timeouts_build_a_working_comms_watchdogs():
    watchdogs = CommsWatchdogs(watchdog_timeouts())
    watchdogs.touch("heartbeat", now_s=0.0)
    assert watchdogs.stale("heartbeat", now_s=0.1) is False
    assert watchdogs.stale("telemetry", now_s=0.1) is True
