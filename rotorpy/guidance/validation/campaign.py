#!/usr/bin/env python
"""Reliability-campaign CLI (plan Phase 9): offline closed-loop trials, not Gazebo.

Explicitly, clearly labelled: every number this script reports comes from the
`rotorpy.guidance.validation` simulation model (a synthetic first-order vehicle-
response model and a synthetic camera built from the project's own pinhole
projection and rendering, see `vehicle_model.py`/`target_world.py`), never from a
Gazebo/PX4/real flight. Real-flight validation has NOT been performed. See the root
`README.md`'s campaign section for the Gazebo validation protocol this script does
not (and cannot, in this environment) run.

Usage::

    python -m rotorpy.guidance.validation.campaign --n-nominal 100 --n-stress 20
    python -m rotorpy.guidance.validation.campaign --quick   # tiny counts, for smoke checks
"""

import argparse
import json
import sys
import time
from typing import List, Optional, Tuple

from rotorpy.guidance.validation.harness import TrialResult, run_trial
from rotorpy.guidance.validation.harness_config import harness_config
from rotorpy.guidance.validation.scenarios import STRESS_FAULT_KINDS, nominal_scenarios, stress_scenarios
from rotorpy.guidance.validation.stats import CampaignReport, aggregate_metrics

DISCLAIMER = (
    "SIMULATION-MODEL RESULTS -- NOT GAZEBO VALIDATION. The vehicle response, camera, "
    "and target motion below are all offline models (rotorpy/guidance/validation/); no "
    "Gazebo, PX4, ROS, or real flight produced them. Real-flight validation has NOT been "
    "performed; see README.md for the Gazebo campaign protocol that remains to be run."
)

# Plan Phase 9's own release-gate wording: at least this many nominal trials.
PROPOSED_MIN_NOMINAL_TRIALS = 100


def _run_stage(specs, config) -> List[TrialResult]:
    return [run_trial(spec, config=config) for spec in specs]


def run_campaign(
    n_nominal: int = PROPOSED_MIN_NOMINAL_TRIALS,
    n_stress: int = len(STRESS_FAULT_KINDS),
    seed: int = 0,
    quick: bool = False,
    config: Optional[dict] = None,
) -> Tuple[CampaignReport, CampaignReport]:
    """Run the nominal and stress stages; return `(nominal_report, stress_report)`.

    `quick=True` (the unit-test/smoke mode) runs a handful of trials instead of the
    real campaign -- the full >=100-nominal-trial run belongs to a manual, offline
    invocation of this module, never inside the automated test suite.
    """
    if quick:
        n_nominal = min(n_nominal, 4)
        n_stress = min(n_stress, len(STRESS_FAULT_KINDS))
    config = harness_config() if config is None else config
    nominal_results = _run_stage(nominal_scenarios(n_nominal, seed=seed), config)
    stress_results = _run_stage(stress_scenarios(n_stress, seed=seed), config)
    return aggregate_metrics("nominal", nominal_results), aggregate_metrics("stress", stress_results)


def format_report(report: CampaignReport) -> str:
    lines = [
        "[{}] trials={} successes={} success_rate={:.3f} wilson95%=({:.3f}, {:.3f})".format(
            report.stage, report.n_trials, report.successes, report.success_rate,
            report.wilson_low, report.wilson_high,
        ),
        "  speed_m_s: mean={} max={}".format(report.mean_speed_m_s, report.max_speed_m_s),
        "  tracking_error_mean_m={}".format(report.mean_tracking_error_m),
        "  visibility_loss_trials={} saturation_trials={}".format(
            report.visibility_loss_trials, report.saturation_trials,
        ),
        "  observation_age_s: mean={} max={}".format(report.mean_observation_age_s, report.max_observation_age_s),
        "  non_finite_detected={}".format(report.non_finite_detected),
        "  failure_categories={}".format(report.failure_categories),
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-nominal", type=int, default=PROPOSED_MIN_NOMINAL_TRIALS)
    parser.add_argument("--n-stress", type=int, default=len(STRESS_FAULT_KINDS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true", help="tiny trial counts; smoke/test use only")
    parser.add_argument("--json", action="store_true", help="also print a machine-readable JSON summary")
    args = parser.parse_args(argv)

    print(DISCLAIMER)
    if not args.quick and args.n_nominal < PROPOSED_MIN_NOMINAL_TRIALS:
        print(
            "warning: --n-nominal {} is below the plan's own {}-trial release-gate "
            "minimum".format(args.n_nominal, PROPOSED_MIN_NOMINAL_TRIALS),
            file=sys.stderr,
        )
    started = time.monotonic()
    nominal_report, stress_report = run_campaign(args.n_nominal, args.n_stress, seed=args.seed, quick=args.quick)
    elapsed_s = time.monotonic() - started
    print(format_report(nominal_report))
    print(format_report(stress_report))
    print("campaign wall time: {:.1f} s".format(elapsed_s))
    if args.json:
        print(json.dumps({
            "disclaimer": DISCLAIMER,
            "nominal": vars(nominal_report),
            "stress": vars(stress_report),
        }, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
