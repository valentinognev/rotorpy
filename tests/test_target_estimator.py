"""Timestamped constant-velocity ground-target motion estimator (plan Phase 6)."""

import numpy as np

from rotorpy.guidance.target_estimator import (
    TargetMotionEstimator,
    TargetMotionStatus,
)

FX = FY = 500.0


def _estimator(**overrides):
    kwargs = dict(
        max_speed_m_s=5.0,
        max_accel_m_s2=2.0,
        outlier_chi2_threshold=9.21,
        max_age_s=1.0,
        max_position_sigma_m=5.0,
    )
    kwargs.update(overrides)
    return TargetMotionEstimator(**kwargs)


def _ray_to(relative_ned):
    """Unit ray direction from a body-origin camera straight at `relative_ned`."""
    relative = np.asarray(relative_ned, dtype=float)
    return relative / np.linalg.norm(relative)


def test_first_observation_initializes_with_zero_velocity_and_observable_status():
    estimator = _estimator()
    estimate = estimator.update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
    )
    assert estimate.status is TargetMotionStatus.OBSERVABLE
    assert np.allclose(estimate.relative_position_ned[:2], [20.0, 0.0], atol=0.5)
    assert np.allclose(estimate.target_velocity_ned, 0.0, atol=1e-9)


def test_stationary_target_does_not_acquire_false_motion_during_a_drone_turn():
    """A fixed world target seen from a maneuvering (turning, accelerating) drone must
    keep an estimated target velocity near zero: the drone's own motion must cancel out
    of the target's own absolute-velocity state, not leak into it."""
    estimator = _estimator()
    target_true = np.array([40.0, 10.0, 15.0])
    vehicle_pos = np.zeros(3)
    dt = 0.05
    rng = np.random.default_rng(7)
    t = 0.0
    last_estimate = None
    for i in range(80):
        # A varying (turning + speeding up) vehicle velocity: drone-induced apparent
        # sightline motion that must not be confused for target motion.
        heading = 0.5 * i * dt
        speed = 3.0 + 2.0 * i * dt
        vehicle_velocity = np.array([speed * np.cos(heading), speed * np.sin(heading), 0.0])
        vehicle_pos = vehicle_pos + vehicle_velocity * dt
        relative_true = target_true - vehicle_pos
        noisy_direction = relative_true + rng.normal(scale=0.01, size=3)
        last_estimate = estimator.update(
            t=t,
            vehicle_velocity_ned=vehicle_velocity,
            ray_origin_ned=np.zeros(3),
            ray_direction_ned=noisy_direction / np.linalg.norm(noisy_direction),
            height_m=15.0,
            reference_height_m=0.0,
            fx=FX,
            fy=FY,
            uncertainty_px=1.0,
        )
        t += dt
    assert last_estimate.status is TargetMotionStatus.OBSERVABLE
    assert np.linalg.norm(last_estimate.target_velocity_ned) < 0.5


def test_moving_target_tracks_constant_velocity_against_synthetic_truth():
    """Synthetic truth is used only inside this test to grade the estimator."""
    estimator = _estimator(max_speed_m_s=10.0)
    target_true = np.array([50.0, 0.0, 15.0])
    true_velocity = np.array([2.0, -1.0, 0.0])
    dt = 0.1
    t = 0.0
    last_estimate = None
    for _ in range(60):
        target_true = target_true + true_velocity * dt
        relative_true = target_true - np.zeros(3)
        last_estimate = estimator.update(
            t=t,
            vehicle_velocity_ned=np.zeros(3),
            ray_origin_ned=np.zeros(3),
            ray_direction_ned=relative_true / np.linalg.norm(relative_true),
            height_m=15.0,
            reference_height_m=0.0,
            fx=FX,
            fy=FY,
            uncertainty_px=1.0,
        )
        t += dt
    assert last_estimate.status is TargetMotionStatus.OBSERVABLE
    assert np.allclose(last_estimate.target_velocity_ned[:2], true_velocity[:2], atol=0.4)


def test_non_finite_inputs_are_rejected_without_corrupting_state():
    estimator = _estimator()
    good = estimator.update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
    )
    rejected = estimator.update(
        t=0.05,
        vehicle_velocity_ned=np.array([float("nan"), 0.0, 0.0]),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
    )
    assert rejected.status is TargetMotionStatus.UNOBSERVABLE
    assert rejected.reason is not None and "non-finite" in rejected.reason
    still = estimator.update(
        t=0.1,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
    )
    assert still.status is TargetMotionStatus.OBSERVABLE
    assert np.allclose(good.relative_position_ned[:2], still.relative_position_ned[:2], atol=1.0)


def test_non_finite_uncertainty_is_rejected_explicitly():
    estimator = _estimator()
    estimate = estimator.update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=float("inf"),
    )
    assert estimate.status is TargetMotionStatus.UNOBSERVABLE
    assert "uncertainty" in estimate.reason


def test_near_horizon_ray_is_flagged_ill_conditioned_with_inflated_uncertainty():
    """A ray that barely reaches the ground plane is ill-conditioned: a small angular
    perturbation sends the ground intersection to infinity (or past the horizon)."""
    estimator = _estimator()
    direction = _ray_to([1000.0, 0.0, 1.0])  # nearly level, grazing incidence
    estimate = estimator.update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=direction,
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=2.0,
    )
    assert estimate.status is TargetMotionStatus.UNOBSERVABLE
    assert "ill-conditioned" in estimate.reason


def test_mahalanobis_outlier_is_rejected_not_absorbed():
    estimator = _estimator()
    for i in range(10):
        estimator.update(
            t=float(i) * 0.1,
            vehicle_velocity_ned=np.zeros(3),
            ray_origin_ned=np.zeros(3),
            ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
            height_m=10.0,
            reference_height_m=0.0,
            fx=FX,
            fy=FY,
            uncertainty_px=1.0,
        )
    outlier = estimator.update(
        t=1.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        # An inconsistent jump (5 m east), but still well-conditioned geometry (not
        # near horizon), so this exercises the Mahalanobis gate, not the ill-
        # conditioned-ray rejection.
        ray_direction_ned=_ray_to([25.0, 5.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
    )
    assert outlier.status is TargetMotionStatus.UNCERTAIN
    assert "outlier" in outlier.reason
    # The rejected measurement must not have dragged the state toward it.
    assert abs(outlier.relative_position_ned[1]) < 1.0


def test_predict_to_compensates_processing_delay_and_reports_closing_speed():
    estimator = _estimator(max_speed_m_s=10.0)
    target_true = np.array([50.0, 0.0, 15.0])
    true_velocity = np.array([2.0, 0.0, 0.0])
    dt = 0.1
    t = 0.0
    for _ in range(40):
        target_true = target_true + true_velocity * dt
        last = estimator.update(
            t=t,
            vehicle_velocity_ned=np.zeros(3),
            ray_origin_ned=np.zeros(3),
            ray_direction_ned=target_true / np.linalg.norm(target_true),
            height_m=15.0,
            reference_height_m=0.0,
            fx=FX,
            fy=FY,
            uncertainty_px=1.0,
        )
        t += dt
    delay = 0.2
    sightline_unit = last.relative_position_ned / np.linalg.norm(last.relative_position_ned)
    predicted = estimator.predict_to(t + delay, np.zeros(3), sightline_unit=sightline_unit)
    assert predicted.status is TargetMotionStatus.OBSERVABLE
    expected_position = last.relative_position_ned[:2] + true_velocity[:2] * delay
    assert np.allclose(predicted.relative_position_ned[:2], expected_position, atol=0.3)
    # Vehicle stationary, target receding (+east... actually +north) at 2 m/s along the
    # sightline: range increasing, so closing speed must be negative.
    assert predicted.closing_speed_m_s < 0.0


def test_stale_prediction_without_fresh_observations_becomes_uncertain():
    estimator = _estimator(max_age_s=0.5)
    estimator.update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
    )
    predicted = estimator.predict_to(5.0, np.zeros(3))
    assert predicted.status is TargetMotionStatus.UNCERTAIN
    assert "stale" in predicted.reason


def test_speed_beyond_the_configured_envelope_is_flagged_uncertain():
    """A fit implying motion far outside the proposed bounded envelope must not be
    treated as an exact, trustworthy estimate."""
    estimator = _estimator(max_speed_m_s=1.0, outlier_chi2_threshold=1e9)
    target_true = np.array([50.0, 0.0, 15.0])
    true_velocity = np.array([8.0, 0.0, 0.0])
    dt = 0.1
    t = 0.0
    last = None
    for _ in range(30):
        target_true = target_true + true_velocity * dt
        last = estimator.update(
            t=t,
            vehicle_velocity_ned=np.zeros(3),
            ray_origin_ned=np.zeros(3),
            ray_direction_ned=target_true / np.linalg.norm(target_true),
            height_m=15.0,
            reference_height_m=0.0,
            fx=FX,
            fy=FY,
            uncertainty_px=1.0,
        )
        t += dt
    assert last.status is TargetMotionStatus.UNCERTAIN
    assert "envelope" in last.reason


def test_predict_to_before_any_observation_is_unobservable():
    estimator = _estimator()
    estimate = estimator.predict_to(1.0, np.zeros(3))
    assert estimate.status is TargetMotionStatus.UNOBSERVABLE
    assert estimate.relative_position_ned is None


def test_reset_drops_state_back_to_unobservable():
    estimator = _estimator()
    estimator.update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
    )
    estimator.reset()
    estimate = estimator.predict_to(0.1, np.zeros(3))
    assert estimate.status is TargetMotionStatus.UNOBSERVABLE


def test_clipped_or_low_confidence_detections_inflate_reported_position_uncertainty():
    """`clipped`/low `confidence` detections must not be trusted as tightly as a clean
    one: both inflate the measurement covariance on top of the geometric Jacobian."""
    clean = _estimator().update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
        clipped=False,
        confidence=1.0,
    )
    clipped = _estimator().update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
        clipped=True,
        confidence=1.0,
    )
    low_confidence = _estimator().update(
        t=0.0,
        vehicle_velocity_ned=np.zeros(3),
        ray_origin_ned=np.zeros(3),
        ray_direction_ned=_ray_to([20.0, 0.0, 10.0]),
        height_m=10.0,
        reference_height_m=0.0,
        fx=FX,
        fy=FY,
        uncertainty_px=1.0,
        clipped=False,
        confidence=0.05,
    )
    assert clipped.position_sigma_m > clean.position_sigma_m
    assert low_confidence.position_sigma_m > clean.position_sigma_m

