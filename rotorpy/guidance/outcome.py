from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional

import numpy as np

from rotorpy.guidance.homing_config import PROPOSED_REACH_RADIUS_M

BOX_CENTER_NED = np.array([100.0, 50.0, -0.5])


@dataclass
class HomingOutcome:
    miss_m: Optional[float]
    timed_out: bool
    time_s: float


def miss_outcome(position_ned: np.ndarray, time_s: float, box_center_ned: np.ndarray = None) -> HomingOutcome:
    center = BOX_CENTER_NED if box_center_ned is None else np.asarray(box_center_ned, dtype=float)
    miss_m = float(np.linalg.norm(np.asarray(position_ned, dtype=float) - center))
    return HomingOutcome(miss_m=miss_m, timed_out=False, time_s=time_s)


def timeout_outcome(time_s: float) -> HomingOutcome:
    return HomingOutcome(miss_m=None, timed_out=True, time_s=time_s)


class RunOutcome(str, Enum):
    """How a flight ended. Values are the stable names written to telemetry."""

    TARGET_REACHED = "TARGET_REACHED"
    GROUND_MISS = "GROUND_MISS"
    TIMEOUT = "TIMEOUT"
    TARGET_LOST = "TARGET_LOST"
    ESTIMATOR_FAILURE = "ESTIMATOR_FAILURE"
    OFFBOARD_FAILURE = "OFFBOARD_FAILURE"
    SIMULATOR_FAILURE = "SIMULATOR_FAILURE"
    STARTUP_FAILURE = "STARTUP_FAILURE"



def classify_run(
    miss_m: Optional[float] = None,
    ground_contact: bool = False,
    timed_out: bool = False,
    target_lost: bool = False,
    estimator_failed: bool = False,
    offboard_failed: bool = False,
    simulator_failed: bool = False,
    startup_failed: bool = False,
    reach_radius_m: float = PROPOSED_REACH_RADIUS_M,
) -> Optional[RunOutcome]:
    """Outcome of a finished run, or None while nothing has ended it.

    `miss_m` is the truth-scored distance to the target. Ground contact alone
    is never a success: TARGET_REACHED needs a finite miss within
    `reach_radius_m`. Infrastructure failures outrank every flight result
    because a run they interrupted proves nothing about homing.
    """
    if startup_failed:
        return RunOutcome.STARTUP_FAILURE
    if simulator_failed:
        return RunOutcome.SIMULATOR_FAILURE
    if offboard_failed:
        return RunOutcome.OFFBOARD_FAILURE
    if estimator_failed:
        return RunOutcome.ESTIMATOR_FAILURE
    if miss_m is not None and math.isfinite(miss_m) and miss_m <= reach_radius_m:
        return RunOutcome.TARGET_REACHED
    if target_lost:
        return RunOutcome.TARGET_LOST
    if ground_contact:
        return RunOutcome.GROUND_MISS
    if timed_out:
        return RunOutcome.TIMEOUT
    return None


def classify_homing_outcome(
    outcome: HomingOutcome, reach_radius_m: float = PROPOSED_REACH_RADIUS_M
) -> Optional[RunOutcome]:
    """Run outcome of a `HomingOutcome` (a ground score or a timeout)."""
    return classify_run(
        miss_m=outcome.miss_m,
        ground_contact=outcome.miss_m is not None,
        timed_out=outcome.timed_out,
        reach_radius_m=reach_radius_m,
    )
