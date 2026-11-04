"""Fixed-seed scenario generation for the reliability campaign (plan Phase 9):
stationary and moving-target nominal trials, and one named fault-injection category
per stress trial, every target-motion parameter bounded by the guidance's own
proposed envelope (`homing_config.PROPOSED_MAX_TARGET_SPEED_M_S`/`_ACCEL_M_S2`).
"""

from typing import List, Optional, Sequence

import numpy as np

from rotorpy.guidance.homing_config import (
    PROPOSED_MAX_TARGET_ACCEL_M_S2,
    PROPOSED_MAX_TARGET_SPEED_M_S,
)
from rotorpy.guidance.validation.faults import FaultSchedule, FaultWindow
from rotorpy.guidance.validation.harness import TrialSpec
from rotorpy.guidance.validation.target_world import TargetMotionProfile

NOMINAL_KINDS = ("stationary", "crossing", "approaching", "receding", "turning", "accelerating")
STRESS_FAULT_KINDS = (
    "camera_dropout", "processing_stall", "stale_telemetry", "pause_reset", "recording_failure",
)
# Safety factor keeping a sampled lateral acceleration (speed * turn_rate) strictly
# inside `TargetMotionProfile.__post_init__`'s envelope bound, not merely at it.
_TURN_RATE_SAFETY = 0.95


def _profile_for_kind(kind: str, rng: np.random.Generator, start_ground_ned: np.ndarray) -> TargetMotionProfile:
    if kind == "stationary":
        return TargetMotionProfile(kind="stationary", start_ground_ned=start_ground_ned)
    speed = float(rng.uniform(0.0, PROPOSED_MAX_TARGET_SPEED_M_S))
    heading = float(rng.uniform(-np.pi, np.pi))
    if kind == "turning":
        max_turn_rate = _TURN_RATE_SAFETY * PROPOSED_MAX_TARGET_ACCEL_M_S2 / max(speed, 0.5)
        turn_rate = float(rng.uniform(-max_turn_rate, max_turn_rate))
        return TargetMotionProfile(
            kind=kind, start_ground_ned=start_ground_ned, speed_m_s=speed,
            heading_rad=heading, turn_rate_rad_s=turn_rate,
        )
    if kind == "accelerating":
        accel = float(rng.uniform(0.0, PROPOSED_MAX_TARGET_ACCEL_M_S2))
        return TargetMotionProfile(
            kind=kind, start_ground_ned=start_ground_ned, speed_m_s=speed,
            heading_rad=heading, accel_m_s2=accel,
        )
    return TargetMotionProfile(kind=kind, start_ground_ned=start_ground_ned, speed_m_s=speed, heading_rad=heading)


def _fault_schedule_for_kind(kind: str, rng: np.random.Generator) -> FaultSchedule:
    onset = float(rng.uniform(2.0, 5.0))
    if kind == "camera_dropout":
        duration = float(rng.uniform(0.6, 1.5))
        return FaultSchedule(camera_dropout=(FaultWindow(onset, onset + duration),))
    if kind == "processing_stall":
        duration = float(rng.uniform(0.3, 1.0))
        return FaultSchedule(processing_stall=(FaultWindow(onset, onset + duration),))
    if kind == "stale_telemetry":
        duration = float(rng.uniform(0.3, 1.0))
        return FaultSchedule(stale_telemetry=(FaultWindow(onset, onset + duration),))
    if kind == "pause_reset":
        return FaultSchedule(pause_reset=(onset,))
    if kind == "recording_failure":
        return FaultSchedule(recording_failure=True)
    return FaultSchedule()


def nominal_scenarios(n: int, seed: int = 0, start_ground_ned: Optional[Sequence[float]] = None) -> List[TrialSpec]:
    """`n` fixed-seed nominal trials (no fault injection), cycling the documented
    stationary/moving-target scenario kinds (`NOMINAL_KINDS`)."""
    start = np.array([100.0, 50.0, 0.0]) if start_ground_ned is None else np.asarray(start_ground_ned, dtype=float)
    specs = []
    for i in range(int(n)):
        trial_seed = int(seed) * 1_000_003 + i
        rng = np.random.default_rng(trial_seed)
        kind = NOMINAL_KINDS[i % len(NOMINAL_KINDS)]
        specs.append(TrialSpec(seed=trial_seed, target_profile=_profile_for_kind(kind, rng, start)))
    return specs


def stress_scenarios(n: int, seed: int = 0, start_ground_ned: Optional[Sequence[float]] = None) -> List[TrialSpec]:
    """`n` fixed-seed stress trials: one named fault-injection category (`STRESS_FAULT_KINDS`)
    per trial, cycled, each on top of a randomly chosen (still envelope-bounded)
    target-motion profile -- a separate stage from `nominal_scenarios`, never mixed
    into the plain success-rate count."""
    start = np.array([100.0, 50.0, 0.0]) if start_ground_ned is None else np.asarray(start_ground_ned, dtype=float)
    specs = []
    for i in range(int(n)):
        trial_seed = int(seed) * 1_000_003 + 500_000 + i
        rng = np.random.default_rng(trial_seed)
        kind = NOMINAL_KINDS[i % len(NOMINAL_KINDS)]
        profile = _profile_for_kind(kind, rng, start)
        fault_kind = STRESS_FAULT_KINDS[i % len(STRESS_FAULT_KINDS)]
        schedule = _fault_schedule_for_kind(fault_kind, rng)
        specs.append(TrialSpec(seed=trial_seed, target_profile=profile, fault_schedule=schedule))
    return specs
