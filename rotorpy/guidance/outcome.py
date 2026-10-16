from dataclasses import dataclass
from typing import Optional

import numpy as np

BOX_CENTER_NED = np.array([100.0, 50.0, -0.5])


@dataclass
class HomingOutcome:
    miss_m: Optional[float]
    timed_out: bool
    time_s: float


def miss_outcome(position_ned: np.ndarray, time_s: float) -> HomingOutcome:
    miss_m = float(np.linalg.norm(np.asarray(position_ned, dtype=float) - BOX_CENTER_NED))
    return HomingOutcome(miss_m=miss_m, timed_out=False, time_s=time_s)


def timeout_outcome(time_s: float) -> HomingOutcome:
    return HomingOutcome(miss_m=None, timed_out=True, time_s=time_s)
