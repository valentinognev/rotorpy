"""Named fault-injection windows for one closed-loop trial (plan Phase 9).

Every fault here is deliberately expressed at the level the real tracker
(`Dockerfiles/camera_homing_tracker.py`) would actually see it, not as a direct
guidance-internal override:

* `camera_dropout` -- both camera images stop arriving (as if the camera topic
  stalled): the harness renders neither frame while a window is active.
* `processing_stall` -- the step loop itself stalls (as if `step()` took too long
  to run, or the consumer thread fell behind): `HomingGuidance.update` is not
  called at all for that tick, same as `command_and_stamp`'s `fresh=False` branch.
* `stale_telemetry` -- the vehicle-state estimate stops updating (as if PX4's own
  estimator or link froze): the harness reports `have_estimate=False` for that
  window, independent of the camera.
* `pause_reset` -- instants (not windows) at which a sim-clock pause/reset/jump is
  simulated by calling `HomingGuidance.reset_derivative_history()` directly, the
  same call `camera_homing_tracker.step()` makes after `VehicleState.
  take_discontinuity()` reports one.
* `recording_failure` -- makes the trial's recording closer raise on
  `RunLifecycle.close()`, exercising the isolated-closer-failure path (plan Phase 8).
"""

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class FaultWindow:
    """A half-open `[start_s, end_s)` fault window, simulation time."""

    start_s: float
    end_s: float

    def active(self, t_s: float) -> bool:
        return self.start_s <= float(t_s) < self.end_s


@dataclass(frozen=True)
class FaultSchedule:
    """Fixed, named fault-injection windows for one closed-loop trial. The empty
    schedule (every field at its default) injects nothing -- a nominal trial."""

    camera_dropout: Tuple[FaultWindow, ...] = ()
    processing_stall: Tuple[FaultWindow, ...] = ()
    stale_telemetry: Tuple[FaultWindow, ...] = ()
    pause_reset: Tuple[float, ...] = ()
    recording_failure: bool = False

    def camera_dropout_active(self, t_s: float) -> bool:
        return any(window.active(t_s) for window in self.camera_dropout)

    def processing_stall_active(self, t_s: float) -> bool:
        return any(window.active(t_s) for window in self.processing_stall)

    def stale_telemetry_active(self, t_s: float) -> bool:
        return any(window.active(t_s) for window in self.stale_telemetry)

    def pause_reset_due(self, t_s: float, tolerance_s: float) -> bool:
        return any(abs(float(t_s) - instant) <= tolerance_s for instant in self.pause_reset)
