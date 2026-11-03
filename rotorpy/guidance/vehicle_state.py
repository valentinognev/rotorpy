"""Timestamp-aware vehicle state built from PX4 telemetry.

PX4 reports attitude (`ATTITUDE_QUATERNION`-style messages) and local
position/velocity (`LOCAL_POSITION_NED`-style messages) on its own clock and
in its own estimator-local frame. This module keeps:

- `ClockModel`: the relationship between that clock (simulation time) and
  wall-clock time, flagging pauses, resets and backwards jumps.
- `AttitudeHistory`: a bounded ring buffer of attitude samples with slerp
  interpolation, so guidance can use the attitude at image-capture time
  instead of whatever the newest sample happens to be.
- `VehicleState`: the aggregate the tracker polls. It converts estimator-
  frame samples into the world frame through an `EstimatorOriginAlignment`
  (coordinates only cross that boundary here, not anywhere upstream or
  downstream), and reports wall-clock ages alongside every sample.

Gazebo ground truth never enters this module: it is PX4 telemetry only.
"""

import math
from dataclasses import dataclass
from typing import Deque, Optional

import numpy as np
from collections import deque
from scipy.spatial.transform import Rotation, Slerp

from rotorpy.guidance.origin_alignment import EstimatorOriginAlignment

# Proposed, not verified: a sim-time regression bigger than this is a discontinuity, not jitter.
PROPOSED_CLOCK_BACKWARD_JUMP_S = 0.05
# Proposed, not verified: wall time advancing this far with no matching sim-time advance means
# the simulation is paused (or badly stalled), not merely between steps.
PROPOSED_CLOCK_PAUSE_WALL_S = 0.5
# Proposed, not verified: how many attitude samples the ring buffer keeps for interpolation.
PROPOSED_ATTITUDE_HISTORY_SAMPLES = 50


class ClockModel(object):
    """Tracks simulation time against wall-clock time across successive samples.

    `update(sim_s, wall_s)` returns one of `None` (no discontinuity),
    `"reset"` (sim time jumped back to near zero -- the simulation
    restarted), `"backward_jump"` (sim time went backwards by more than
    `backward_jump_s`, but not to zero) or `"paused"` (wall time advanced by
    at least `pause_wall_s` while sim time did not advance). Consumers such
    as `AttitudeHistory` should clear buffered history on any non-`None`
    result: the sim-time axis is no longer contiguous with what came before.
    """

    def __init__(
        self,
        backward_jump_s: float = PROPOSED_CLOCK_BACKWARD_JUMP_S,
        pause_wall_s: float = PROPOSED_CLOCK_PAUSE_WALL_S,
    ) -> None:
        self.backward_jump_s = float(backward_jump_s)
        self.pause_wall_s = float(pause_wall_s)
        self._last_sim_s: Optional[float] = None
        self._last_wall_s: Optional[float] = None

    def update(self, sim_s: float, wall_s: float) -> Optional[str]:
        sim_s = float(sim_s)
        wall_s = float(wall_s)
        if self._last_sim_s is None:
            self._last_sim_s, self._last_wall_s = sim_s, wall_s
            return None
        sim_delta = sim_s - self._last_sim_s
        wall_delta = wall_s - self._last_wall_s
        discontinuity = None
        if sim_delta < -self.backward_jump_s:
            discontinuity = "reset" if abs(sim_s) <= self.backward_jump_s else "backward_jump"
        elif wall_delta >= self.pause_wall_s and sim_delta <= self.backward_jump_s:
            discontinuity = "paused"
        self._last_sim_s, self._last_wall_s = sim_s, wall_s
        return discontinuity


class AttitudeHistory(object):
    """Bounded ring buffer of (sim_s, attitude quaternion xyzw) samples with slerp.

    `interpolate(sim_s)` returns the slerp-interpolated attitude and whether
    `sim_s` fell inside the buffered window. Outside the window it still
    returns the nearest endpoint's attitude (useful as a last resort) but
    flags `in_range=False` so callers can reject or discount a capture time
    that has aged out of the buffer.
    """

    def __init__(self, maxlen: int = PROPOSED_ATTITUDE_HISTORY_SAMPLES) -> None:
        self._times: Deque[float] = deque(maxlen=maxlen)
        self._quats: Deque[np.ndarray] = deque(maxlen=maxlen)

    def __len__(self) -> int:
        return len(self._times)

    def record(self, sim_s: float, quat_xyzw: np.ndarray) -> None:
        self._times.append(float(sim_s))
        self._quats.append(np.array(quat_xyzw, dtype=float, copy=True))

    def clear(self) -> None:
        self._times.clear()
        self._quats.clear()

    def latest(self):
        """The most recently recorded attitude, or `None` if nothing has been recorded."""
        return None if not self._quats else np.array(self._quats[-1], copy=True)

    def interpolate(self, sim_s: float):
        if not self._times:
            return None, False
        times = list(self._times)
        sim_s = float(sim_s)
        if len(times) == 1:
            return np.array(self._quats[0], copy=True), math.isclose(sim_s, times[0], abs_tol=1e-9)
        in_range = times[0] <= sim_s <= times[-1]
        clamped = min(max(sim_s, times[0]), times[-1])
        rotations = Rotation.from_quat(np.array(self._quats))
        slerp = Slerp(times, rotations)
        quat = slerp([clamped]).as_quat()[0]
        return quat, in_range


@dataclass
class VehicleSample:
    """One timestamp-aware read of `VehicleState`. All vectors are world-frame NED."""

    attitude_quat_world: Optional[np.ndarray]
    attitude_in_range: bool
    attitude_age_s: Optional[float]
    position_ned: Optional[np.ndarray]
    velocity_ned: Optional[np.ndarray]
    height_above_ground_m: Optional[float]
    position_age_s: Optional[float]
    velocity_age_s: Optional[float]
    discontinuity: Optional[str]


class VehicleState(object):
    """PX4 attitude and local-position telemetry, timestamp-aware, in the world frame."""

    def __init__(
        self,
        alignment: EstimatorOriginAlignment,
        attitude_history_samples: int = PROPOSED_ATTITUDE_HISTORY_SAMPLES,
        clock_backward_jump_s: float = PROPOSED_CLOCK_BACKWARD_JUMP_S,
        clock_pause_wall_s: float = PROPOSED_CLOCK_PAUSE_WALL_S,
    ) -> None:
        self._alignment = alignment
        self._clock = ClockModel(backward_jump_s=clock_backward_jump_s, pause_wall_s=clock_pause_wall_s)
        self._attitude_history = AttitudeHistory(maxlen=attitude_history_samples)
        self._attitude_wall_s: Optional[float] = None
        self._position_world_ned: Optional[np.ndarray] = None
        self._velocity_world_ned: Optional[np.ndarray] = None
        self._position_wall_s: Optional[float] = None
        self.last_discontinuity: Optional[str] = None

    def ingest_attitude(self, sim_s: float, wall_s: float, quat_estimator_xyzw) -> None:
        """Record one PX4 `ATTITUDE_QUATERNION`-style sample (estimator-frame, FRD body)."""
        self.last_discontinuity = self._clock.update(sim_s, wall_s)
        if self.last_discontinuity is not None:
            self._attitude_history.clear()
        world_quat = self._alignment.attitude_to_world(quat_estimator_xyzw)
        self._attitude_history.record(sim_s, world_quat)
        self._attitude_wall_s = float(wall_s)

    def ingest_local_position(self, wall_s: float, position_estimator_ned, velocity_estimator_ned) -> None:
        """Record one PX4 `LOCAL_POSITION_NED`-style sample (estimator-local frame)."""
        self._position_world_ned = self._alignment.estimator_to_world(position_estimator_ned)
        self._velocity_world_ned = self._alignment.estimator_vector_to_world(velocity_estimator_ned)
        self._position_wall_s = float(wall_s)

    def sample_at(self, capture_sim_s: Optional[float], now_wall_s: float) -> VehicleSample:
        """The vehicle state for an image captured at `capture_sim_s` (or `None` for "now",
        i.e. no fresh image this tick: report the most recent attitude as-is)."""
        if capture_sim_s is not None:
            quat, in_range = self._attitude_history.interpolate(capture_sim_s)
        else:
            quat = self._attitude_history.latest()
            in_range = quat is not None
        now_wall_s = float(now_wall_s)
        attitude_age = None if self._attitude_wall_s is None else max(0.0, now_wall_s - self._attitude_wall_s)
        position_age = None if self._position_wall_s is None else max(0.0, now_wall_s - self._position_wall_s)
        height = (
            None
            if self._position_world_ned is None
            else self._alignment.height_above_ground_world(self._position_world_ned)
        )
        return VehicleSample(
            attitude_quat_world=quat,
            attitude_in_range=in_range,
            attitude_age_s=attitude_age,
            position_ned=self._position_world_ned,
            velocity_ned=self._velocity_world_ned,
            height_above_ground_m=height,
            position_age_s=position_age,
            velocity_age_s=position_age,
            discontinuity=self.last_discontinuity,
        )
