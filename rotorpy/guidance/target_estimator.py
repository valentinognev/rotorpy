"""Timestamped constant-velocity ground-target motion estimator (plan Phase 6).

Pure module: no Gazebo truth, no ROS/MAVLink. Consumes one camera ray per tick (plus
its pixel uncertainty, clip flag and confidence) together with the vehicle's own
velocity at capture time -- never the target's truth position -- and tracks the
target's own world-aligned ground velocity with a constant-velocity Kalman filter.
`predict_to` advances that filter to a later "control time" (processing-delay
compensation) and reports relative velocity / closing speed.

Why this cancels drone-induced image motion instead of mistaking it for target
motion: the filter's measurement is a ground-plane relative position (vehicle to
target, from `sightline.reference_relative_position`), and the *known* vehicle
velocity enters the predict step as an exact control input (not noise). For a truly
stationary target, the relative position drifts by exactly `-vehicle_velocity * dt`
regardless of how the drone turns or accelerates; the filter attributes that whole
drift to the known vehicle motion, leaving the estimated target velocity at zero
(see `tests/test_target_estimator.py::test_stationary_target_does_not_acquire_false_motion_during_a_drone_turn`).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from rotorpy.guidance.sightline import reference_relative_position

# State is [rn, re, vn, ve]: relative (vehicle-to-target) north/east position, and the
# target's own world-aligned north/east velocity. The down component of the relative
# position is never filtered -- the Task-2 ground-plane geometry already gives it
# directly, deterministically, from the vehicle's own height each tick -- so this
# module only reports the latest raw down value alongside the filtered horizontal
# state (a documented limitation: `predict_to` does not advance the down component).
_H = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
_STATE_DIM = 4


class TargetMotionStatus(Enum):
    #: No trustworthy estimate at all: never observed yet, or this control time's
    #: geometry/inputs are rejected and no earlier state exists.
    UNOBSERVABLE = "unobservable"
    #: An estimate exists but its uncertainty, age, or implied speed exceeds the
    #: configured bound: callers must not treat it as exact (fall back instead).
    UNCERTAIN = "uncertain"
    #: A fresh, well-conditioned, in-envelope estimate.
    OBSERVABLE = "observable"


@dataclass(frozen=True)
class TargetMotionEstimate:
    """One estimate, at (or predicted to) a requested time.

    `relative_position_ned`/`relative_velocity_ned` are vehicle-to-target, NED
    world-aligned axes (not body). `closing_speed_m_s` is the *vehicle's* closing
    rate on the target along the unit sightline (positive = range decreasing),
    matching `navigation.pn_acceleration`'s stationary-target convention, generalized
    to a moving target: ``dot(sightline_unit, vehicle_velocity - target_velocity)``.
    Every caller must check `status`: anything other than `OBSERVABLE` means this is
    not to be treated as an exact position/velocity (see homing.py wiring, which falls
    back to the existing bearing-only/pure-pursuit steering whenever this is not
    `OBSERVABLE`).
    """

    status: TargetMotionStatus
    reason: Optional[str] = None
    relative_position_ned: Optional[np.ndarray] = None
    relative_velocity_ned: Optional[np.ndarray] = None
    target_velocity_ned: Optional[np.ndarray] = None
    closing_speed_m_s: Optional[float] = None
    position_sigma_m: Optional[float] = None
    velocity_sigma_m_s: Optional[float] = None
    age_s: Optional[float] = None


def _all_finite(*values) -> bool:
    for value in values:
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            return False
    return True


def _scalar_sigma(cov_2x2: np.ndarray) -> float:
    """RMS-ish one-number summary of a 2x2 covariance block (not a full ellipse)."""
    return float(np.sqrt(max(0.0, 0.5 * (cov_2x2[0, 0] + cov_2x2[1, 1]))))


def _tangent_basis(direction: np.ndarray):
    """Two unit vectors orthogonal to `direction` (itself assumed unit)."""
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(direction, reference))) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])
    e1 = np.cross(direction, reference)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    return e1, e2


class TargetMotionEstimator:
    """Constant-velocity Kalman filter over the target's own ground velocity.

    No built-in literal defaults (mirrors `tracking.TrackingStateMachine`): every
    threshold is a required constructor argument, with the actual proposed numbers
    named in `homing_config.target_motion_thresholds()`.
    """

    def __init__(
        self,
        max_speed_m_s: float,
        max_accel_m_s2: float,
        outlier_chi2_threshold: float,
        max_age_s: float,
        max_position_sigma_m: float,
        clipped_inflation: float = 3.0,
        low_confidence_inflation: float = 2.0,
        confidence_floor: float = 0.2,
    ) -> None:
        self.max_speed_m_s = float(max_speed_m_s)
        self.max_accel_m_s2 = float(max_accel_m_s2)
        self.outlier_chi2_threshold = float(outlier_chi2_threshold)
        self.max_age_s = float(max_age_s)
        self.max_position_sigma_m = float(max_position_sigma_m)
        self.clipped_inflation = float(clipped_inflation)
        self.low_confidence_inflation = float(low_confidence_inflation)
        self.confidence_floor = float(confidence_floor)
        self.reset()

    def reset(self) -> None:
        """Drop all filter state: the next `update` starts a fresh track."""
        self._state: Optional[np.ndarray] = None
        self._cov: Optional[np.ndarray] = None
        self._filter_t: Optional[float] = None
        self._last_good_t: Optional[float] = None
        self._last_relative_down_m: Optional[float] = None

    @property
    def initialized(self) -> bool:
        return self._state is not None

    # --- measurement geometry -------------------------------------------------

    def _measurement(
        self,
        ray_origin_ned,
        ray_direction_ned,
        height_m: float,
        reference_height_m: float,
        fx: float,
        fy: float,
        uncertainty_px: float,
        clipped: bool,
        confidence: float,
    ):
        """(measurement[3], R[2x2], reason) or (None, None, reason) when unusable."""
        if not _all_finite(ray_origin_ned, ray_direction_ned, height_m, reference_height_m, fx, fy):
            return None, None, "non-finite ray/geometry input rejected"
        if not (np.isfinite(uncertainty_px) and uncertainty_px > 0.0):
            return None, None, "invalid pixel uncertainty rejected"
        origin = np.asarray(ray_origin_ned, dtype=float)
        direction = np.asarray(ray_direction_ned, dtype=float)
        norm = float(np.linalg.norm(direction))
        if norm == 0.0:
            return None, None, "zero-length ray rejected"
        direction = direction / norm
        relative = reference_relative_position(height_m, origin, direction, reference_height_m)
        if relative is None:
            return None, None, "ray does not reach the target-reference plane"

        avg_f = 0.5 * (float(fx) + float(fy))
        sigma_rad = max(float(uncertainty_px) / avg_f, 1e-6)
        e1, e2 = _tangent_basis(direction)
        jacobian_columns = []
        for tangent in (e1, e2):
            plus = direction + sigma_rad * tangent
            minus = direction - sigma_rad * tangent
            plus = plus / np.linalg.norm(plus)
            minus = minus / np.linalg.norm(minus)
            relative_plus = reference_relative_position(height_m, origin, plus, reference_height_m)
            relative_minus = reference_relative_position(height_m, origin, minus, reference_height_m)
            if relative_plus is None or relative_minus is None:
                return None, None, (
                    "ill-conditioned: a one-sigma ray perturbation leaves the "
                    "target-reference plane (near-horizon or severe clipping)"
                )
            jacobian_columns.append((relative_plus[:2] - relative_minus[:2]) / (2.0 * sigma_rad))
        jacobian = np.column_stack(jacobian_columns)
        R = sigma_rad * sigma_rad * (jacobian @ jacobian.T)

        inflation = 1.0
        if clipped:
            inflation *= self.clipped_inflation
        if confidence < self.confidence_floor:
            inflation *= self.low_confidence_inflation
        R = R * inflation
        if not np.all(np.isfinite(R)) or _scalar_sigma(R) > self.max_position_sigma_m:
            return None, None, (
                "ill-conditioned: propagated position uncertainty exceeds the "
                "configured bound"
            )
        return relative, R, None

    # --- filter mechanics ------------------------------------------------------

    def _predict_matrices(self, dt: float, vehicle_velocity_ned: np.ndarray):
        F = np.eye(_STATE_DIM)
        F[0, 2] = dt
        F[1, 3] = dt
        control = np.zeros(_STATE_DIM)
        control[0] = -float(vehicle_velocity_ned[0]) * dt
        control[1] = -float(vehicle_velocity_ned[1]) * dt
        sigma_a = self.max_accel_m_s2
        q = sigma_a * sigma_a
        Q = np.zeros((_STATE_DIM, _STATE_DIM))
        for i, j in ((0, 2), (1, 3)):
            Q[i, i] = q * dt ** 4 / 4.0
            Q[i, j] = Q[j, i] = q * dt ** 3 / 2.0
            Q[j, j] = q * dt ** 2
        return F, control, Q

    def _predict(self, dt: float, vehicle_velocity_ned: np.ndarray):
        """Predicted (state, covariance) `dt` seconds ahead; does not mutate `self`."""
        if dt <= 0.0:
            return self._state.copy(), self._cov.copy()
        F, control, Q = self._predict_matrices(dt, vehicle_velocity_ned)
        state = F @ self._state + control
        cov = F @ self._cov @ F.T + Q
        return state, cov

    def _status_for(self, t: float, state: np.ndarray, cov: np.ndarray):
        if self._last_good_t is None:
            return TargetMotionStatus.UNOBSERVABLE, "no accepted observation yet"
        age = t - self._last_good_t
        if age > self.max_age_s:
            return TargetMotionStatus.UNCERTAIN, "stale: age {:.2f}s exceeds max_age_s".format(age)
        position_sigma = _scalar_sigma(cov[:2, :2])
        if position_sigma > self.max_position_sigma_m:
            return TargetMotionStatus.UNCERTAIN, (
                "position sigma {:.2f} m exceeds max_position_sigma_m".format(position_sigma)
            )
        speed = float(np.hypot(state[2], state[3]))
        if speed > self.max_speed_m_s:
            return TargetMotionStatus.UNCERTAIN, (
                "estimated target speed {:.2f} m/s exceeds the proposed bounded-motion "
                "envelope".format(speed)
            )
        return TargetMotionStatus.OBSERVABLE, None

    def _estimate(
        self, t: float, state: np.ndarray, cov: np.ndarray,
        vehicle_velocity_ned: np.ndarray, sightline_unit: Optional[np.ndarray],
        override_reason: Optional[str] = None,
        override_status: Optional[TargetMotionStatus] = None,
    ) -> TargetMotionEstimate:
        status, reason = self._status_for(t, state, cov)
        if override_status is not None:
            status = override_status
        reason = override_reason if override_reason is not None else reason
        down = 0.0 if self._last_relative_down_m is None else self._last_relative_down_m
        relative_position = np.array([state[0], state[1], down])
        target_velocity = np.array([state[2], state[3], 0.0])
        vehicle_velocity = np.asarray(vehicle_velocity_ned, dtype=float)
        relative_velocity = np.array([
            target_velocity[0] - vehicle_velocity[0],
            target_velocity[1] - vehicle_velocity[1],
            0.0,
        ])
        closing_speed = None
        if sightline_unit is not None:
            unit = np.asarray(sightline_unit, dtype=float)
            unit = unit / np.linalg.norm(unit)
            closing_speed = float(np.dot(unit, vehicle_velocity - target_velocity))
        return TargetMotionEstimate(
            status=status,
            reason=reason,
            relative_position_ned=relative_position,
            relative_velocity_ned=relative_velocity,
            target_velocity_ned=target_velocity,
            closing_speed_m_s=closing_speed,
            position_sigma_m=_scalar_sigma(cov[:2, :2]),
            velocity_sigma_m_s=_scalar_sigma(cov[2:, 2:]),
            age_s=None if self._last_good_t is None else t - self._last_good_t,
        )

    @staticmethod
    def _unobservable(reason: str) -> TargetMotionEstimate:
        return TargetMotionEstimate(status=TargetMotionStatus.UNOBSERVABLE, reason=reason)

    # --- public API --------------------------------------------------------

    def update(
        self,
        t: float,
        vehicle_velocity_ned,
        ray_origin_ned,
        ray_direction_ned,
        height_m: float,
        reference_height_m: float,
        fx: float,
        fy: float,
        uncertainty_px: float,
        clipped: bool = False,
        confidence: float = 1.0,
        sightline_unit: Optional[np.ndarray] = None,
    ) -> TargetMotionEstimate:
        """Advance the filter to `t` and fold in one ray observation.

        Non-finite `t`/`vehicle_velocity_ned` abort the whole call (nothing is
        mutated): the predict step itself needs them. A non-finite or ill-conditioned
        ray/pixel-uncertainty input still lets time (and the filter's own
        uncertainty) advance -- only the correction is skipped -- since the vehicle's
        own motion is still known and valid.
        """
        t = float(t)
        vehicle_velocity = np.asarray(vehicle_velocity_ned, dtype=float)
        if not _all_finite(t, vehicle_velocity):
            return self._unobservable("non-finite time/vehicle-velocity input rejected")
        if self._filter_t is not None and t < self._filter_t:
            return self._unobservable("observation time precedes the filter's own time")

        measurement, R, reason = self._measurement(
            ray_origin_ned, ray_direction_ned, height_m, reference_height_m,
            fx, fy, uncertainty_px, clipped, confidence,
        )

        if self._state is None:
            if measurement is None:
                return self._unobservable(reason or "no usable observation yet")
            self._state = np.array([measurement[0], measurement[1], 0.0, 0.0])
            self._cov = np.diag([
                R[0, 0], R[1, 1],
                self.max_speed_m_s ** 2, self.max_speed_m_s ** 2,
            ])
            self._filter_t = t
            self._last_good_t = t
            self._last_relative_down_m = float(measurement[2])
            return self._estimate(t, self._state, self._cov, vehicle_velocity, sightline_unit)

        dt = t - self._filter_t
        state_pred, cov_pred = self._predict(dt, vehicle_velocity)

        if measurement is None:
            self._state, self._cov, self._filter_t = state_pred, cov_pred, t
            return self._estimate(t, state_pred, cov_pred, vehicle_velocity, sightline_unit)

        z = measurement[:2]
        y = z - _H @ state_pred
        S = _H @ cov_pred @ _H.T + R
        mahalanobis2 = float(y.T @ np.linalg.solve(S, y))
        if mahalanobis2 > self.outlier_chi2_threshold:
            self._state, self._cov, self._filter_t = state_pred, cov_pred, t
            return self._estimate(
                t, state_pred, cov_pred, vehicle_velocity, sightline_unit,
                override_reason="rejected: Mahalanobis outlier (d2={:.1f})".format(mahalanobis2),
                override_status=TargetMotionStatus.UNCERTAIN,
            )

        K = cov_pred @ _H.T @ np.linalg.inv(S)
        state_new = state_pred + K @ y
        cov_new = (np.eye(_STATE_DIM) - K @ _H) @ cov_pred
        self._state, self._cov, self._filter_t = state_new, cov_new, t
        self._last_good_t = t
        self._last_relative_down_m = float(measurement[2])
        return self._estimate(t, state_new, cov_new, vehicle_velocity, sightline_unit)

    def predict_to(
        self, t_control: float, vehicle_velocity_ned, sightline_unit: Optional[np.ndarray] = None,
    ) -> TargetMotionEstimate:
        """The estimate at `t_control`, without mutating the stored filter state.

        `vehicle_velocity_ned` is the vehicle's current (not historical) velocity:
        the predict gap is the short processing delay this call compensates for, so
        holding velocity constant over it is the same bounded assumption the rest of
        the estimator already makes between ticks.
        """
        t_control = float(t_control)
        vehicle_velocity = np.asarray(vehicle_velocity_ned, dtype=float)
        if self._state is None:
            return self._unobservable("no accepted observation yet")
        if not _all_finite(t_control, vehicle_velocity):
            return self._unobservable("non-finite time/vehicle-velocity input rejected")
        dt = t_control - self._filter_t
        if dt < 0.0:
            dt = 0.0
        state_pred, cov_pred = self._predict(dt, vehicle_velocity)
        return self._estimate(t_control, state_pred, cov_pred, vehicle_velocity, sightline_unit)
