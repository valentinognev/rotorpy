"""Closed-loop flights of the attitude and body-rate command modes through `Px4Cascade`.

Slow (about 40 s per flight), so every test is `homing_e2e` and the default suite skips them.
Bounds sit above the measured values (attitude and rate both miss by 2.4 m, peak descent
2.9 m/s, peak speed 10 m/s) so only a real regression trips them.
"""

import functools

import numpy as np
import pytest

from rotorpy.guidance.homing_config import load_homing_config
from rotorpy.guidance.runner import run_recorded_homing

pytestmark = pytest.mark.homing_e2e

MAX_MISS_M = 4.0
MAX_DESCENT_M_S = 4.0
MAX_SPEED_M_S = 12.0
MAX_MODE_DISAGREEMENT_M = 0.5
MAX_TAU_SPREAD_M = 1.0


@functools.lru_cache(maxsize=None)
def _flight(mode, track_tau_s=0.5):
    config = load_homing_config("examples/camera_homing.jsonc") | {
        "controller": "pid",
        "command_mode": mode,
        "track_tau_s": track_tau_s,
    }
    outcome, record = run_recorded_homing(config=config)
    velocity = np.diff(np.asarray(record.position), axis=0) / np.diff(record.time)[:, None]
    return outcome, velocity


@pytest.mark.parametrize("mode", ["attitude", "rate"])
def test_angle_mode_homes_within_the_measured_envelope(mode):
    outcome, velocity = _flight(mode)
    assert not outcome.timed_out
    assert outcome.miss_m <= MAX_MISS_M
    assert -velocity[:, 2].min() <= MAX_DESCENT_M_S  # record positions are z-up
    assert np.linalg.norm(velocity, axis=1).max() <= MAX_SPEED_M_S


def test_rate_mode_flies_like_attitude_mode():
    attitude, _ = _flight("attitude")
    rate, _ = _flight("rate")
    assert abs(attitude.miss_m - rate.miss_m) <= MAX_MODE_DISAGREEMENT_M


def test_default_track_tau_sits_on_a_plateau():
    misses = [_flight("attitude", tau)[0].miss_m for tau in (0.3, 0.5, 0.8)]
    assert None not in misses
    assert max(misses) - min(misses) <= MAX_TAU_SPREAD_M
