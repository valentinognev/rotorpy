"""Wilson confidence intervals and aggregate campaign statistics (plan Phase 9).

Release gate (plan line ~385): "Report speed, tracking error, visibility loss,
latency, saturation, and success rate together" and "Report confidence intervals
and failure categories; do not equate a small successful batch with proven
reliability." `aggregate_metrics` is the one place that bundle is assembled.
"""

import math
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.validation.harness import TrialResult

# The standard two-sided 95% z-score.
DEFAULT_Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int, z: float = DEFAULT_Z_95) -> Tuple[float, float]:
    """Wilson score confidence interval for a binomial proportion, clamped to
    `[0, 1]`. `total = 0` reports the uninformative `(0.0, 1.0)` instead of
    dividing by zero -- a campaign stage that ran no trials has no evidence
    either way, not perfect confidence in anything."""
    if total <= 0:
        return 0.0, 1.0
    if successes < 0 or successes > total:
        raise ValueError("successes must be between 0 and total")
    n = float(total)
    p_hat = successes / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = p_hat + z2 / (2.0 * n)
    margin = z * math.sqrt((p_hat * (1.0 - p_hat) + z2 / (4.0 * n)) / n)
    low = (center - margin) / denom
    high = (center + margin) / denom
    return max(0.0, low), min(1.0, high)


@dataclass
class CampaignReport:
    """Speed, tracking error, visibility loss, latency, saturation, and success
    rate together for one campaign stage ("nominal" or "stress"), plus the Wilson
    interval and failure-category counts -- never a bare success count alone."""

    stage: str
    n_trials: int
    successes: int
    success_rate: float
    wilson_low: float
    wilson_high: float
    failure_categories: Dict[str, int]
    mean_speed_m_s: Optional[float]
    max_speed_m_s: Optional[float]
    mean_tracking_error_m: Optional[float]
    visibility_loss_trials: int
    mean_observation_age_s: Optional[float]
    max_observation_age_s: Optional[float]
    saturation_trials: int
    non_finite_detected: bool


def aggregate_metrics(stage: str, results: Sequence[TrialResult]) -> CampaignReport:
    n = len(results)
    successes = sum(1 for r in results if r.outcome == RunOutcome.TARGET_REACHED)
    categories = Counter(r.outcome.value for r in results if r.outcome != RunOutcome.TARGET_REACHED)
    low, high = wilson_interval(successes, n)
    speeds = [r.mean_speed_m_s for r in results]
    max_speeds = [r.max_speed_m_s for r in results]
    tracking_errors = [r.mean_tracking_error_m for r in results if r.mean_tracking_error_m is not None]
    ages = [r.mean_observation_age_s for r in results if r.mean_observation_age_s is not None]
    max_ages = [r.max_observation_age_s for r in results if r.max_observation_age_s is not None]
    return CampaignReport(
        stage=stage,
        n_trials=n,
        successes=successes,
        success_rate=(successes / n) if n else 0.0,
        wilson_low=low,
        wilson_high=high,
        failure_categories=dict(categories),
        mean_speed_m_s=(sum(speeds) / n) if n else None,
        max_speed_m_s=max(max_speeds) if max_speeds else None,
        mean_tracking_error_m=(sum(tracking_errors) / len(tracking_errors)) if tracking_errors else None,
        visibility_loss_trials=sum(1 for r in results if r.visibility_loss_episodes > 0),
        mean_observation_age_s=(sum(ages) / len(ages)) if ages else None,
        max_observation_age_s=max(max_ages) if max_ages else None,
        saturation_trials=sum(1 for r in results if r.saturation_ticks > 0),
        non_finite_detected=any(r.non_finite_detected for r in results),
    )
