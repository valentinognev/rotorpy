"""Truth-side vehicle-response model for the offline reliability harness (plan Phase 9).

Not a flight-dynamics model and not Gazebo: a deliberately simple first-order
velocity-tracking response. It reuses the project's own tilt/upright-specific-force
law (`rotorpy.guidance.homing.track_acceleration`, the same function the Python
example's `flat_output` uses to drive `SE3Control`) so the simulated vehicle cannot
instantaneously snap to a commanded velocity or bank past its configured tilt limit,
and adds two effects the production Gazebo/PX4 path has that a pure kinematic model
does not: command latency (the velocity/yaw goal the model tracks is whichever
transmitted command was issued `command_latency_s` ago, not whatever was most
recently sent) and a short pose history so the synthetic camera can deliberately
render a stale ("captured", not "now") pose -- `image_delay_s`.

Truth-side only: never imported by `rotorpy.guidance.homing`/`tracking`/
`target_estimator`/`command_limits` or any other guidance-side module. Its outputs
(`position_ned`, `velocity_ned`, `attitude_sim_quat`) are exactly the ground truth
the harness's synthetic camera renders from and the truth scorer consumes -- never
fed to `HomingGuidance` as anything other than the vehicle's own state, the same
role Gazebo's PX4 telemetry plays for the real tracker.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.guidance.frames import ned_to_sim, sim_to_ned
from rotorpy.guidance.homing import track_acceleration

_GRAVITY = 9.81


def _wrap_to_pi(angle_rad: float) -> float:
    """`angle_rad` folded into `(-pi, pi]`."""
    return float(np.arctan2(np.sin(angle_rad), np.cos(angle_rad)))


def attitude_from_specific_force(accel_sim: np.ndarray, yaw_rad: float) -> np.ndarray:
    """Homing-sim attitude quaternion whose body +z follows the specific force.

    Mirrors `rotorpy.controllers.quadrotor_control.control.SE3Control.update`'s own
    `b3_des`/`c1_des`/`b2_des`/`b1_des` construction (FLU-ENU, thrust axis +1): body
    +z tracks `accel_sim + [0, 0, g]` (the thrust direction a quadrotor would need to
    produce that acceleration while supporting its own weight) and the heading
    constrains the remaining degree of freedom about that axis. Falls back to a
    level, yaw-only attitude when the specific force is degenerate (zero, or
    parallel to the heading -- a near-vertical dive/climb), rather than producing a
    NaN quaternion from a zero-length cross product.
    """
    specific_force = np.asarray(accel_sim, dtype=float) + np.array([0.0, 0.0, _GRAVITY])
    norm = float(np.linalg.norm(specific_force))
    if not np.isfinite(norm) or norm < 1e-9:
        return Rotation.from_euler("z", yaw_rad).as_quat()
    b3 = specific_force / norm
    heading = np.array([np.cos(yaw_rad), np.sin(yaw_rad), 0.0])
    b2_raw = np.cross(b3, heading)
    b2_norm = float(np.linalg.norm(b2_raw))
    if b2_norm < 1e-6:
        return Rotation.from_euler("z", yaw_rad).as_quat()
    b2 = b2_raw / b2_norm
    b1 = np.cross(b2, b3)
    rotation_matrix = np.stack([b1, b2, b3]).T
    return Rotation.from_matrix(rotation_matrix).as_quat()


@dataclass
class VehiclePoseSample:
    """One recorded instant of vehicle truth, for `VehicleResponseModel.pose_at`."""

    t_s: float
    position_ned: np.ndarray
    velocity_ned: np.ndarray
    attitude_sim_quat: np.ndarray


class VehicleResponseModel(object):
    """First-order velocity/yaw-tracking truth vehicle, with latency and a pose history.

    Every threshold is a required constructor argument (same convention as
    `tracking.TrackingStateMachine`/`target_estimator.TargetMotionEstimator`): no
    built-in literal defaults here, so a caller cannot silently diverge from the
    vehicle envelope the guidance it drives was configured with.
    """

    def __init__(
        self,
        position_ned,
        velocity_ned,
        yaw_rad: float,
        tau_s: float,
        max_accel_m_s2: float,
        max_tilt_deg: float,
        min_specific_up_g: float,
        max_yaw_rate_rad_s: float,
        command_latency_s: float = 0.0,
        image_delay_s: float = 0.0,
        pose_history_len: int = 4000,
    ) -> None:
        self.position_ned = np.array(position_ned, dtype=float, copy=True)
        self.velocity_ned = np.array(velocity_ned, dtype=float, copy=True)
        self.yaw_rad = _wrap_to_pi(float(yaw_rad))
        self._tau_s = float(tau_s)
        self._max_accel_m_s2 = float(max_accel_m_s2)
        self._max_tilt_deg = float(max_tilt_deg)
        self._min_specific_up_g = float(min_specific_up_g)
        self._max_yaw_rate_rad_s = float(max_yaw_rate_rad_s)
        self._command_latency_s = float(command_latency_s)
        self._image_delay_s = float(image_delay_s)
        self._goal_velocity_ned = np.array(self.velocity_ned, copy=True)
        self._goal_yaw_rad = self.yaw_rad
        self._pending: Deque[Tuple[float, np.ndarray, float]] = deque()
        self.attitude_sim_quat = attitude_from_specific_force(np.zeros(3), self.yaw_rad)
        self._history: Deque[VehiclePoseSample] = deque(maxlen=int(pose_history_len))
        self._history.append(VehiclePoseSample(
            0.0, self.position_ned.copy(), self.velocity_ned.copy(), self.attitude_sim_quat.copy()
        ))

    def enqueue_command(self, t_s: float, velocity_ned, yaw_rad: float) -> None:
        """Queue a newly transmitted command, delayed by `command_latency_s`.

        Non-finite input is dropped outright (never queued): a vehicle truth state
        must stay finite regardless of what a fault-injected or buggy command tried
        to send, the same "never propagate NaN/inf" rule `command_limits.limit_command`
        enforces on the guidance side.
        """
        velocity = np.asarray(velocity_ned, dtype=float)
        yaw = float(yaw_rad)
        if not (np.all(np.isfinite(velocity)) and np.isfinite(yaw)):
            return
        self._pending.append((float(t_s) + self._command_latency_s, velocity.copy(), yaw))

    def _advance_goal(self, t_s: float) -> None:
        while self._pending and self._pending[0][0] <= t_s:
            _, velocity_ned, yaw_rad = self._pending.popleft()
            self._goal_velocity_ned = velocity_ned
            self._goal_yaw_rad = yaw_rad

    def step(self, dt_s: float, t_s: float) -> None:
        """Advance the truth state by `dt_s`, ending at sim time `t_s`."""
        dt_s = float(dt_s)
        self._advance_goal(float(t_s))
        velocity_sim = ned_to_sim(self.velocity_ned)
        goal_sim = ned_to_sim(self._goal_velocity_ned)
        accel_sim = track_acceleration(
            velocity_sim, goal_sim, np.zeros(3),
            tau_s=self._tau_s, max_tilt_deg=self._max_tilt_deg, min_specific_up_g=self._min_specific_up_g,
        )
        norm = float(np.linalg.norm(accel_sim))
        if norm > self._max_accel_m_s2 and norm > 0.0:
            accel_sim = accel_sim * (self._max_accel_m_s2 / norm)
        velocity_sim = velocity_sim + accel_sim * dt_s
        position_sim = ned_to_sim(self.position_ned) + velocity_sim * dt_s
        self.velocity_ned = sim_to_ned(velocity_sim)
        self.position_ned = sim_to_ned(position_sim)
        yaw_err = _wrap_to_pi(self._goal_yaw_rad - self.yaw_rad)
        max_step = abs(self._max_yaw_rate_rad_s) * dt_s
        self.yaw_rad = _wrap_to_pi(self.yaw_rad + float(np.clip(yaw_err, -max_step, max_step)))
        self.attitude_sim_quat = attitude_from_specific_force(accel_sim, self.yaw_rad)
        self._history.append(VehiclePoseSample(
            float(t_s), self.position_ned.copy(), self.velocity_ned.copy(), self.attitude_sim_quat.copy()
        ))

    def pose_at(self, capture_t_s: float) -> VehiclePoseSample:
        """The last recorded pose at or before `capture_t_s`.

        No interpolation (a documented simplification): the history is sampled at
        the same `dt_s` the model is stepped at, so the nearest-before sample is at
        most one physics tick stale. Before the first recorded sample (or with an
        empty history), the current state stands in for it.
        """
        if not self._history:
            return VehiclePoseSample(
                float(capture_t_s), self.position_ned.copy(), self.velocity_ned.copy(), self.attitude_sim_quat.copy()
            )
        best = self._history[0]
        for sample in self._history:
            if sample.t_s > capture_t_s:
                break
            best = sample
        return best

    def delayed_pose(self, now_t_s: float) -> VehiclePoseSample:
        """The pose the synthetic camera should render: `image_delay_s` stale."""
        return self.pose_at(float(now_t_s) - self._image_delay_s)
