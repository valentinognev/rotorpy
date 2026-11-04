"""Offline closed-loop reliability harness and campaign tooling (plan Phase 9).

Everything under this package is truth-side: it builds a synthetic vehicle and
target, drives the real guidance-side modules (`HomingGuidance`, its tracking
state machine and target-motion estimator, `command_limits`, `SetpointStream`)
through them, and scores the result with the truth-side `scoring.TruthScorer`.
Nothing here is imported by any guidance/estimator module -- the directory sits
outside `tests/test_homing_truth_boundary.py`'s (deliberately non-recursive)
`rotorpy/guidance/*.py` glob, and `tests/test_homing_validation_truth_boundary.py`
pins that choice down so it stays deliberate rather than incidental.

No Gazebo, PX4, ROS, or real flight is used anywhere in this package: the
vehicle response, the camera images, and the target motion are all models,
clearly documented as proposed/unverified. See `README.md`/`UPDATES.md` for the
reproducible validation commands and the Gazebo campaign protocol this offline
harness cannot itself run.
"""
