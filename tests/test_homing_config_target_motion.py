"""Proposed target-motion envelope and estimator thresholds live in homing_config,
named and documented (plan Phase 6)."""

from rotorpy.guidance.homing_config import (
    PROPOSED_MAX_TARGET_ACCEL_M_S2,
    PROPOSED_MAX_TARGET_AGE_S,
    PROPOSED_MAX_TARGET_POSITION_SIGMA_M,
    PROPOSED_MAX_TARGET_SPEED_M_S,
    PROPOSED_TARGET_CLIPPED_INFLATION,
    PROPOSED_TARGET_CONFIDENCE_FLOOR,
    PROPOSED_TARGET_LOW_CONFIDENCE_INFLATION,
    PROPOSED_TARGET_OUTLIER_CHI2_THRESHOLD,
    target_motion_thresholds,
)
from rotorpy.guidance.target_estimator import TargetMotionEstimator, TargetMotionStatus


def test_target_motion_thresholds_are_positive_and_named():
    thresholds = target_motion_thresholds()
    assert thresholds == {
        "max_speed_m_s": PROPOSED_MAX_TARGET_SPEED_M_S,
        "max_accel_m_s2": PROPOSED_MAX_TARGET_ACCEL_M_S2,
        "outlier_chi2_threshold": PROPOSED_TARGET_OUTLIER_CHI2_THRESHOLD,
        "max_age_s": PROPOSED_MAX_TARGET_AGE_S,
        "max_position_sigma_m": PROPOSED_MAX_TARGET_POSITION_SIGMA_M,
        "clipped_inflation": PROPOSED_TARGET_CLIPPED_INFLATION,
        "low_confidence_inflation": PROPOSED_TARGET_LOW_CONFIDENCE_INFLATION,
        "confidence_floor": PROPOSED_TARGET_CONFIDENCE_FLOOR,
    }
    assert all(value > 0.0 for value in thresholds.values())


def test_target_motion_thresholds_build_a_working_estimator():
    estimator = TargetMotionEstimator(**target_motion_thresholds())
    estimate = estimator.update(
        t=0.0,
        vehicle_velocity_ned=[0.0, 0.0, 0.0],
        ray_origin_ned=[0.0, 0.0, 0.0],
        ray_direction_ned=[0.0, 0.0, 1.0],
        height_m=10.0,
        reference_height_m=0.0,
        fx=500.0,
        fy=500.0,
        uncertainty_px=1.0,
    )
    assert estimate.status is TargetMotionStatus.OBSERVABLE
