"""Truth-side scoring of a homing run: target contact, ground contact, miss distance.

This module is the only place Gazebo (or simulator) *truth* about the vehicle and the
target is combined. The guidance and estimator modules never import it -- that boundary
is enforced by `tests/test_homing_truth_boundary.py` -- so truth cannot leak into the
homing law. Everything here is NED, metres and seconds, with no ROS dependency.

Scoring model:

* The target is a vertical cylinder standing on a (possibly moving) ground contact
  point: it fills `z in [-height_m, 0]` relative to that point (NED, up is negative).
* Vehicle and target truth are separate timestamped traces. The target is interpolated
  to every vehicle sample time, so a moving target is scored at matching timestamps.
* Between two consecutive matched samples both bodies are taken as moving linearly, so
  the relative motion is a straight segment; the closest approach of that segment to the
  cylinder is exact (the point-to-cylinder distance is convex along a segment). A
  high-speed pass cannot disappear between samples. Collision events from the simulator
  (`note_collision`) are accepted as a second, independent source of contact.
* Target contact (the vehicle touching the cylinder) is separate from ground contact.
  Ground contact alone is never `TARGET_REACHED`; the outcome comes from
  `outcome.classify_run` with the truth miss distance.
"""

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Iterable, Optional, Sequence, Tuple

import numpy as np

from rotorpy.guidance.homing_config import PROPOSED_REACH_RADIUS_M
from rotorpy.guidance.outcome import RunOutcome, classify_run

# Proposed, not verified: vehicle-to-cylinder-surface clearance that counts as touching
# the target (roughly the Hummingbird's half extent plus propeller guard).
PROPOSED_CONTACT_MARGIN_M = 0.3
# Proposed, not verified: height of the vehicle's reference point above the ground at
# which it is on the ground (the Gazebo body rests a few centimetres above z = 0).
PROPOSED_GROUND_CLEARANCE_M = 0.25
# Proposed, not verified, tightened in fix round 1 (was 1.0 s): a target truth sample
# older than this at a vehicle sample time is not used for scoring (the pairing is
# counted as unmatched instead of extrapolated/held stale). A moving target can cover
# several metres in even this bound; it existed mainly to tolerate normal publish
# jitter, not to paper over a target truth source that stalled or was never sampled.
PROPOSED_MAX_TARGET_GAP_S = 0.25
# Proposed, not verified (finding 2, fix round 1): how recent -- relative to the run's
# terminal event (ground contact, timeout, or the last sample) -- the single closest
# surface approach must be for proximity alone (no actual contact) to decide
# TARGET_REACHED. Without this bound the *minimum* distance over the whole flight would
# decide the outcome, so an early close fly-over followed by a crash somewhere else
# entirely would still count as reached. Actual `target_contact` is itself a terminal
# event (segment-vs-cylinder contact or a collision event) and is never subject to this
# bound.
PROPOSED_REACH_WINDOW_S = 2.0
# Bound on the target trace the online scorer keeps (oldest dropped first).
PROPOSED_TARGET_TRACE_LEN = 4096

_SEARCH_ITERATIONS = 80
_ENTRY_ITERATIONS = 50
_GOLDEN = (math.sqrt(5.0) - 1.0) / 2.0

_COLLISION_KINDS = ("target", "ground")


@dataclass(frozen=True)
class TargetShape:
    """A standing vertical cylinder: radius and height, metres."""

    radius_m: float
    height_m: float


def distance_to_cylinder(point_rel_ned, shape: TargetShape) -> float:
    """Distance from a point (relative to the cylinder's ground contact, NED) to its solid."""
    x, y, z = (float(value) for value in point_rel_ned)
    radial = max(math.hypot(x, y) - shape.radius_m, 0.0)
    axial = max(-shape.height_m - z, z, 0.0)
    return math.hypot(radial, axial)


def _along(p0: np.ndarray, p1: np.ndarray, fraction: float) -> np.ndarray:
    return p0 + fraction * (p1 - p0)


def segment_closest_approach(p0_rel_ned, p1_rel_ned, shape: TargetShape) -> Tuple[float, float]:
    """Minimum distance to the cylinder along a straight segment, and where on it (0..1).

    The distance to a convex set is convex along a line, so a golden-section search on
    the segment finds the true minimum (a plateau of zero distance returns any point of it).
    """
    p0 = np.asarray(p0_rel_ned, dtype=float)
    p1 = np.asarray(p1_rel_ned, dtype=float)
    low, high = 0.0, 1.0
    a = high - _GOLDEN * (high - low)
    b = low + _GOLDEN * (high - low)
    fa = distance_to_cylinder(_along(p0, p1, a), shape)
    fb = distance_to_cylinder(_along(p0, p1, b), shape)
    for _ in range(_SEARCH_ITERATIONS):
        if fa <= fb:
            high, b, fb = b, a, fa
            a = high - _GOLDEN * (high - low)
            fa = distance_to_cylinder(_along(p0, p1, a), shape)
        else:
            low, a, fa = a, b, fb
            b = low + _GOLDEN * (high - low)
            fb = distance_to_cylinder(_along(p0, p1, b), shape)
    best = (low + high) / 2.0
    candidates = [(distance_to_cylinder(_along(p0, p1, s), shape), s) for s in (0.0, best, 1.0)]
    return min(candidates)


def _first_within(p0: np.ndarray, p1: np.ndarray, shape: TargetShape, margin: float, closest: float) -> float:
    """Earliest fraction on the segment at or before `closest` whose distance is <= margin."""
    if distance_to_cylinder(p0, shape) <= margin:
        return 0.0
    low, high = 0.0, closest
    for _ in range(_ENTRY_ITERATIONS):
        middle = (low + high) / 2.0
        if distance_to_cylinder(_along(p0, p1, middle), shape) <= margin:
            high = middle
        else:
            low = middle
    return high


class TraceBuffer(object):
    """A bounded, strictly time-ordered trace of timestamped 3-vectors."""

    def __init__(self, maxlen: int = PROPOSED_TARGET_TRACE_LEN) -> None:
        self._samples: Deque[Tuple[float, np.ndarray]] = deque(maxlen=int(maxlen))
        self.rejected = 0

    def __len__(self) -> int:
        return len(self._samples)

    def append(self, t_s: float, value) -> bool:
        """Add a sample. Non-finite or non-increasing samples are counted and refused."""
        t_s = float(t_s)
        vector = np.asarray(value, dtype=float).reshape(3)
        if not (math.isfinite(t_s) and np.all(np.isfinite(vector))):
            self.rejected += 1
            return False
        if self._samples and t_s <= self._samples[-1][0]:
            self.rejected += 1
            return False
        self._samples.append((t_s, vector))
        return True

    def at(self, t_s: float, max_gap_s: float) -> Optional[np.ndarray]:
        """Linear interpolation at `t_s`, or None when no sample is within `max_gap_s` of it
        on the relevant side (this never extrapolates beyond the newest sample's gap)."""
        if not self._samples:
            return None
        t_s = float(t_s)
        first_t, first_v = self._samples[0]
        last_t, last_v = self._samples[-1]
        if t_s < first_t:
            return first_v.copy() if first_t - t_s <= max_gap_s else None
        if t_s >= last_t:
            return last_v.copy() if t_s - last_t <= max_gap_s else None
        previous_t, previous_v = self._samples[0]
        for sample_t, sample_v in self._samples:
            if sample_t >= t_s:
                if sample_t == t_s:
                    return sample_v.copy()
                weight = (t_s - previous_t) / (sample_t - previous_t)
                return previous_v + weight * (sample_v - previous_v)
            previous_t, previous_v = sample_t, sample_v
        return last_v.copy()

    def last(self) -> Optional[Tuple[float, np.ndarray]]:
        """The most recent raw sample as-is (`None` if empty), with no gap bound: for
        best-effort reporting (`ground_miss_m`) only, never used to decide a match or a
        score -- `at()`'s gap bound is what guards those."""
        if not self._samples:
            return None
        t_s, vector = self._samples[-1]
        return t_s, vector.copy()


@dataclass(frozen=True)
class ScoreResult:
    """What the truth says about a run so far."""

    samples: int
    matched_samples: int
    unmatched_samples: int
    min_surface_distance_m: Optional[float]
    min_distance_time_s: Optional[float]
    target_contact: bool
    contact_time_s: Optional[float]
    ground_contact: bool
    ground_time_s: Optional[float]
    timed_out: bool
    elapsed_s: Optional[float]
    # Fix round 1 (finding 2): the horizontal ("ground-projected") distance from the
    # vehicle to the target's ground-contact point at the run's terminal sample (ground
    # impact, the timeout-triggering sample, or else the last sample seen) -- distinct
    # from `min_surface_distance_m`, which is the closest 3-D approach to the cylinder
    # anywhere in the flight.
    ground_miss_m: Optional[float]
    # Fix round 1 (minor b): at least one vehicle sample had no usable target truth at
    # its time (the target trace was empty with no static fallback, or every sample was
    # outside `max_target_gap_s`). Such samples are never scored as reached; this flag
    # only surfaces that the run (or part of it) was not fully scorable.
    target_truth_unavailable: bool


class TruthScorer(object):
    """Online truth scorer. Feed target samples, then vehicle samples; read `result()`.

    `add_vehicle_sample`/`note_collision` return the termination status
    (`"target"`, `"ground"` or `"timeout"`) the first time a run-ending condition is met,
    and `None` otherwise (including every call after the run has ended).
    """

    def __init__(
        self,
        shape: TargetShape,
        timeout_s: float,
        contact_margin_m: float = PROPOSED_CONTACT_MARGIN_M,
        reach_radius_m: float = PROPOSED_REACH_RADIUS_M,
        ground_clearance_m: float = PROPOSED_GROUND_CLEARANCE_M,
        ground_down_m: float = 0.0,
        max_target_gap_s: float = PROPOSED_MAX_TARGET_GAP_S,
        static_target_ground_ned: Optional[Sequence[float]] = None,
        target_trace_len: int = PROPOSED_TARGET_TRACE_LEN,
        reach_window_s: float = PROPOSED_REACH_WINDOW_S,
    ) -> None:
        self.shape = shape
        self.timeout_s = float(timeout_s)
        self.contact_margin_m = float(contact_margin_m)
        self.reach_radius_m = float(reach_radius_m)
        self.ground_clearance_m = float(ground_clearance_m)
        self.ground_down_m = float(ground_down_m)
        self.max_target_gap_s = float(max_target_gap_s)
        self.reach_window_s = float(reach_window_s)
        self._static = None if static_target_ground_ned is None else np.asarray(static_target_ground_ned, dtype=float)
        self._target = TraceBuffer(target_trace_len)
        self._t0: Optional[float] = None
        self._previous: Optional[Tuple[float, np.ndarray, np.ndarray]] = None
        self._last_sample: Optional[Tuple[float, np.ndarray]] = None
        self._samples = 0
        self._matched = 0
        self._unmatched = 0
        self._min_distance: Optional[float] = None
        self._min_time: Optional[float] = None
        self._target_contact = False
        self._contact_time: Optional[float] = None
        self._ground_contact = False
        self._ground_time: Optional[float] = None
        self._timed_out = False
        self._elapsed: Optional[float] = None
        self._ended = False

    @property
    def ended(self) -> bool:
        return self._ended

    def add_target_sample(self, t_s: float, ground_contact_ned) -> bool:
        return self._target.append(t_s, ground_contact_ned)

    def _target_at(self, t_s: float) -> Optional[np.ndarray]:
        if len(self._target) == 0:
            return None if self._static is None else self._static.copy()
        return self._target.at(t_s, self.max_target_gap_s)

    def _target_reference(self, t_s: float) -> Optional[np.ndarray]:
        """Best-effort target position for *reporting* (`ground_miss_m`) only -- unlike
        `_target_at`, falls back to the most recent raw sample even when it is stale, so
        a diagnostic miss distance is still recorded instead of silently omitted. Never
        used to decide a match or a score."""
        value = self._target_at(t_s)
        if value is not None:
            return value
        last = self._target.last()
        return None if last is None else last[1]

    def _ground_projected_miss(self) -> Optional[float]:
        """Horizontal distance from the vehicle to the target's ground-contact point at
        the run's terminal sample (ground impact, the timeout-triggering sample, or else
        the last sample seen so far)."""
        if self._last_sample is None:
            return None
        t_s, vehicle = self._last_sample
        target = self._target_reference(t_s)
        if target is None:
            return None
        return float(math.hypot(vehicle[0] - target[0], vehicle[1] - target[1]))

    def add_vehicle_sample(self, t_s: float, position_ned) -> Optional[str]:
        t_s = float(t_s)
        vehicle = np.asarray(position_ned, dtype=float).reshape(3)
        if not (math.isfinite(t_s) and np.all(np.isfinite(vehicle))) or self._ended:
            return None
        if self._previous is not None and t_s <= self._previous[0]:
            return None
        self._samples += 1
        self._last_sample = (t_s, vehicle.copy())
        if self._t0 is None:
            self._t0 = t_s
        self._elapsed = t_s - self._t0
        target = self._target_at(t_s)
        status = None
        if target is None:
            self._unmatched += 1
            self._previous = None
        else:
            self._matched += 1
            status = self._score_segment(t_s, vehicle, target)
        if status is None and -(vehicle[2] - self.ground_down_m) <= self.ground_clearance_m:
            self._ground_contact = True
            self._ground_time = t_s
            status = "ground"
        if status is None and self._elapsed >= self.timeout_s:
            self._timed_out = True
            status = "timeout"
        if status is not None:
            self._ended = True
        return status

    def _score_segment(self, t_s: float, vehicle: np.ndarray, target: np.ndarray) -> Optional[str]:
        relative = vehicle - target
        previous = self._previous
        self._previous = (t_s, vehicle, target)
        if previous is None:
            distance, fraction, p0 = distance_to_cylinder(relative, self.shape), 1.0, relative
        else:
            p0 = previous[1] - previous[2]
            distance, fraction = segment_closest_approach(p0, relative, self.shape)
        if self._min_distance is None or distance < self._min_distance:
            self._min_distance = distance
            start_t = t_s if previous is None else previous[0]
            self._min_time = start_t + fraction * (t_s - start_t)
        if distance <= self.contact_margin_m:
            if previous is None:
                contact_t = t_s
            else:
                entry = _first_within(p0, relative, self.shape, self.contact_margin_m, fraction)
                contact_t = previous[0] + entry * (t_s - previous[0])
            self._mark_target_contact(contact_t)
            return "target"
        return None

    def _mark_target_contact(self, t_s: float) -> None:
        self._target_contact = True
        if self._contact_time is None or t_s < self._contact_time:
            self._contact_time = t_s
        self._ended = True

    def note_collision(self, t_s: float, kind: str) -> Optional[str]:
        """A simulator collision event: `"target"` or `"ground"` (anything else is ignored)."""
        if kind not in _COLLISION_KINDS or not math.isfinite(float(t_s)):
            return None
        first = not self._ended
        if kind == "target":
            self._mark_target_contact(float(t_s))
        else:
            self._ground_contact = True
            if self._ground_time is None:
                self._ground_time = float(t_s)
            self._ended = True
        return kind if first else None

    def _recent_enough(self) -> bool:
        """Whether the closest surface approach happened within `reach_window_s` of the
        most recent sample seen (the run's terminal tick: ground impact, the
        timeout-triggering sample, or else the last sample so far)."""
        if self._min_time is None or self._last_sample is None:
            return False
        return (self._last_sample[0] - self._min_time) <= self.reach_window_s

    def result(self) -> ScoreResult:
        return ScoreResult(
            samples=self._samples,
            matched_samples=self._matched,
            unmatched_samples=self._unmatched,
            min_surface_distance_m=self._min_distance,
            min_distance_time_s=self._min_time,
            target_contact=self._target_contact,
            contact_time_s=self._contact_time,
            ground_contact=self._ground_contact,
            ground_time_s=self._ground_time,
            timed_out=self._timed_out,
            elapsed_s=self._elapsed,
            ground_miss_m=self._ground_projected_miss(),
            target_truth_unavailable=self._unmatched > 0,
        )

    def miss_m(self) -> Optional[float]:
        """Truth miss distance (to the cylinder surface); 0 when contact was reported."""
        return 0.0 if self._target_contact else self._min_distance

    def _reach_eligible(self) -> bool:
        """TARGET_REACHED-by-proximity criterion (finding 2, fix round 1): actual contact
        with the target body always qualifies (it is itself a terminal event); a mere
        closest-surface-approach within `reach_radius_m` only qualifies when it happened
        within `reach_window_s` of the run's terminal tick -- so a close fly-over long
        before a later, distant ground impact does not retroactively count."""
        if self._target_contact:
            return True
        return (
            self._min_distance is not None
            and self._min_distance <= self.reach_radius_m
            and self._recent_enough()
        )

    def outcome(self, **failure_flags) -> Optional[RunOutcome]:
        """The run outcome once something ended it, else None.

        `failure_flags` pass straight to `classify_run` (`target_lost`, `startup_failed`,
        ...): infrastructure failures outrank the truth score there.
        """
        if not (self._ended or any(failure_flags.values())):
            return None
        reach_eligible = self._reach_eligible()
        classify_miss = 0.0 if self._target_contact else (self._min_distance if reach_eligible else None)
        return classify_run(
            miss_m=classify_miss,
            ground_contact=self._ground_contact,
            timed_out=self._timed_out,
            reach_radius_m=self.reach_radius_m,
            **failure_flags,
        )


def score_traces(
    vehicle: Iterable[Tuple[float, Sequence[float]]],
    target: Iterable[Tuple[float, Sequence[float]]],
    shape: TargetShape,
    contact_margin_m: float = PROPOSED_CONTACT_MARGIN_M,
    timeout_s: float = math.inf,
    **kwargs,
) -> ScoreResult:
    """Score complete recorded traces (the batch twin of the online `TruthScorer`)."""
    target = list(target)
    scorer = TruthScorer(
        shape, timeout_s=timeout_s, contact_margin_m=contact_margin_m,
        target_trace_len=max(len(target), 1), **kwargs,
    )
    for t_s, ground in target:
        scorer.add_target_sample(t_s, ground)
    for t_s, position in vehicle:
        scorer.add_vehicle_sample(t_s, position)
    return scorer.result()
