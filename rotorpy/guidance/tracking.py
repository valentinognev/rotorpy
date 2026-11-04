"""Explicit, recoverable per-target tracking state machine.

`STARTUP -> ACQUIRE -> TRACK -> TEMPORARY_LOSS -> REACQUIRE`, with two
terminal states: `ABORT` (the guidance's own explicit failure response to a
prolonged loss or runaway prediction uncertainty) and `COMPLETE` (never set
internally; only a caller -- the tracker/runner, driven by the truth-side
scorer -- calls `complete()`, since guidance itself never reads truth).

This module knows nothing about cameras, pixels or geometry: it consumes one
`CameraSignal` per named camera per tick and only decides *which* camera (if
any) should drive steering this tick, and whether the run must stop. The
geometry/steering law in `homing.py` is the only caller.

Replaces the previous ad hoc `_upward_latched`/`_lost`/`_permanent_loss`
bookkeeping in `HomingGuidance`, which had no recovery path once the
upward camera's one-way latch lost visibility: a single later-recorded
flight showed the forward camera still seeing the target for the rest of
the flight while guidance had already given up for good (see
`tests/test_homing_recorded_handoff_replay.py`).
"""

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional, Sequence


class TrackingState(Enum):
    STARTUP = "STARTUP"
    ACQUIRE = "ACQUIRE"
    TRACK = "TRACK"
    TEMPORARY_LOSS = "TEMPORARY_LOSS"
    REACQUIRE = "REACQUIRE"
    ABORT = "ABORT"
    COMPLETE = "COMPLETE"


_TERMINAL_STATES = (TrackingState.ABORT, TrackingState.COMPLETE)


@dataclass(frozen=True)
class CameraSignal:
    """One camera's per-tick target signal, already adapted from whatever perception
    this camera uses (structured `pixels.TargetObservation` confidence/uncertainty on
    the Gazebo/mounts path, or a plain visible/not-visible flag on the legacy path).

    `confidence` is only compared between simultaneously visible cameras (it is
    meaningless, and never read, while `visible` is False). `uncertainty_px` is the
    camera's own per-camera track's growing prediction uncertainty (for example
    `pixels.TargetAssociator.track.sigma_px`); `None` when this camera keeps no such
    measure (the legacy path), which never contributes to the uncertainty bound.
    """

    visible: bool
    confidence: float = 0.0
    uncertainty_px: Optional[float] = None


class TrackingStateMachine(object):
    """One target track across camera changes, with bounded recovery and one explicit,
    one-way failure response instead of indefinitely holding a stale command.

    No built-in literal defaults here (mirrors `watchdog.CommsWatchdogs`): every
    threshold is a required constructor argument, and `rotorpy.guidance.homing_config`
    is the one place that names the actual proposed numbers
    (`PROPOSED_ACQUIRE_FRAMES`, `PROPOSED_REACQUIRE_FRAMES`,
    `PROPOSED_CAMERA_SWITCH_MARGIN`, `PROPOSED_MAX_PREDICTION_GAP_S`,
    `PROPOSED_MAX_TRACK_UNCERTAINTY_PX`). `HomingGuidance` keeps its own private
    literal defaults (matching those proposed numbers) only so direct construction in
    tests keeps working without naming every threshold.
    """

    def __init__(
        self,
        acquire_frames: int,
        reacquire_frames: int,
        switch_margin: float,
        max_prediction_gap_s: float,
        max_uncertainty_px: float,
        camera_priority: Sequence[str] = (),
    ) -> None:
        self._acquire_frames = max(1, int(acquire_frames))
        self._reacquire_frames = max(1, int(reacquire_frames))
        self._switch_margin = float(switch_margin)
        self._max_gap_s = float(max_prediction_gap_s)
        self._max_uncertainty_px = float(max_uncertainty_px)
        self._priority = {name: rank for rank, name in enumerate(camera_priority)}
        self.state = TrackingState.STARTUP
        # The camera actively driving (or last driving, through a loss) steering.
        # `None` only before the first-ever acquisition.
        self.camera: Optional[str] = None
        self.abort_reason: Optional[str] = None
        self.complete_reason: Optional[str] = None
        self.discontinuous = False
        self._consecutive = 0
        self._lost_since: Optional[float] = None
        self._force_discontinuity = False

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES

    def update(self, t: float, signals: Mapping[str, CameraSignal],
               max_prediction_gap_s: Optional[float] = None) -> None:
        """Advance one tick. No-op once terminal (`ABORT`/`COMPLETE` are one-way).

        `max_prediction_gap_s` optionally overrides the constructor's gap bound for this
        tick only (the uncertainty bound is unaffected). `homing.py` raises it during a
        low-altitude final approach that was holding a confident lock, so dead reckoning
        can coast across the brief nadir blind spot between the two cameras instead of
        aborting the moment the target drops out of both. `None` keeps the constructor
        default, so every other caller is unchanged.
        """
        if self.is_terminal:
            self.discontinuous = False
            return
        t = float(t)
        visible = {name: signal for name, signal in signals.items() if signal.visible}
        chosen = self._select_camera(visible)
        if chosen is not None:
            self._on_visible(t, chosen)
            return
        self._on_invisible(t, signals, max_prediction_gap_s)

    def complete(self, reason: Optional[str] = None) -> None:
        """External, truth-scored "the flight is over" signal (Task 8's scorer, or a
        runner's own ground/timeout check) -- never set from inside this module, and
        never from a truth value guidance itself read. One-way: does nothing once
        already terminal, so a late call cannot resurrect or override `ABORT`."""
        if self.is_terminal:
            return
        self.state = TrackingState.COMPLETE
        self.complete_reason = reason
        self.discontinuous = False

    def force_discontinuity(self) -> None:
        """Flag the next successful observation as discontinuous and forget any
        in-progress prediction gap, after an external clock discontinuity: the elapsed
        time this module would otherwise compute back to `_lost_since` spans a jump
        that never really elapsed, so it must not count toward `max_prediction_gap_s`
        (and must not survive to fabricate a derivative in `homing.py`'s steering law).
        Does not touch `state`: a clock jump is not evidence the target is lost or
        found, and must not revive a terminal `ABORT`/`COMPLETE`.
        """
        self._lost_since = None
        self._force_discontinuity = True

    def _on_visible(self, t: float, chosen: str) -> None:
        switched = chosen != self.camera
        if switched:
            self.camera = chosen
            self._consecutive = 0
        self._consecutive += 1
        self._lost_since = None
        if self.state in (TrackingState.STARTUP, TrackingState.ACQUIRE):
            confirmed = self._consecutive >= self._acquire_frames
            self.state = TrackingState.TRACK if confirmed else TrackingState.ACQUIRE
            natural_break = confirmed
        elif self.state in (TrackingState.TEMPORARY_LOSS, TrackingState.REACQUIRE):
            confirmed = self._consecutive >= self._reacquire_frames
            self.state = TrackingState.TRACK if confirmed else TrackingState.REACQUIRE
            natural_break = confirmed
        else:  # already TRACK
            natural_break = switched
        # A pending `force_discontinuity()` (external clock discontinuity) always wins,
        # even on a tick that would not otherwise count as a break in its own right.
        self.discontinuous = natural_break or self._force_discontinuity
        self._force_discontinuity = False

    def _on_invisible(self, t: float, signals: Mapping[str, CameraSignal],
                      max_prediction_gap_s: Optional[float] = None) -> None:
        self._consecutive = 0
        self.discontinuous = False
        if self.state is TrackingState.STARTUP:
            return
        if self.state is TrackingState.ACQUIRE:
            # Never confirmed: a flicker before acquisition is routine, not a loss.
            self.state = TrackingState.STARTUP
            self.camera = None
            return
        if self._lost_since is None:
            self._lost_since = t
        self.state = TrackingState.TEMPORARY_LOSS
        gap_s = t - self._lost_since
        max_gap_s = self._max_gap_s if max_prediction_gap_s is None else float(max_prediction_gap_s)
        uncertainty = None if self.camera is None else signals.get(self.camera)
        uncertainty_px = None if uncertainty is None else uncertainty.uncertainty_px
        if gap_s > max_gap_s:
            self._abort("max_prediction_gap_exceeded")
        elif uncertainty_px is not None and uncertainty_px > self._max_uncertainty_px:
            self._abort("max_track_uncertainty_exceeded")

    def _abort(self, reason: str) -> None:
        self.state = TrackingState.ABORT
        self.abort_reason = reason
        self.discontinuous = False

    def _select_camera(self, visible: Mapping[str, CameraSignal]) -> Optional[str]:
        """The camera to steer from this tick, or `None` when nothing is visible.

        Ranked by `camera_priority` first (lower index wins; a Gazebo camera not in
        the priority sequence ranks last), confidence second. Switching away from the
        still-visible active camera to a same-or-lower-priority rival needs that
        rival's confidence to clear the active camera's by `switch_margin`
        (hysteresis: a single marginal competing detection must not steal the track).
        A strictly higher-priority camera pre-empts immediately (matches the
        documented forward-then-upward design), and losing the active camera's own
        visibility always falls back immediately to whatever else is visible (no
        hysteresis protects an already-lost camera).
        """
        if not visible:
            return None

        def rank(name: str) -> int:
            return self._priority.get(name, len(self._priority))

        best = min(visible, key=lambda name: (rank(name), -visible[name].confidence))
        if self.camera is not None and self.camera in visible and self.camera != best:
            active_rank, best_rank = rank(self.camera), rank(best)
            if active_rank <= best_rank:
                active_confidence = visible[self.camera].confidence
                if visible[best].confidence <= active_confidence + self._switch_margin:
                    return self.camera
        return best
