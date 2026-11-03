"""Explicit recoverable vs fatal error classes for the tracker, and their run outcomes.

The tracker used to catch every `step()` exception and carry on. Now an error is one of:

* `RecoverableError` -- a transient problem (for example one Gazebo service call failing).
  The step is skipped, the error is counted, and a `RecoverableBudget` escalates it to a
  fatal `SimulatorFailure` once it keeps happening (bounded retries).
* `FatalError` subclasses -- end the run: normal setpoint transmission stops (the
  `SetpointStream` failure latch) and exactly one outcome is written from `outcome_flags`.
* Anything else is unexpected, so it is fatal too (`outcome_flags` maps it to an
  infrastructure `SIMULATOR_FAILURE` with an `unexpected:` reason). Continuing after an
  unknown error would hide it.
"""

from typing import Dict, Optional

from rotorpy.guidance.homing_config import (
    PROPOSED_RECOVERABLE_ERROR_MAX_CONSECUTIVE,
    PROPOSED_RECOVERABLE_ERROR_MAX_TOTAL,
)


class RunError(Exception):
    """Base class of every deliberate tracker error."""


class RecoverableError(RunError):
    """A transient failure: skip this step, count it, retry within a bound."""


class ServiceUnavailable(RecoverableError, RuntimeError):
    """A Gazebo/ROS service call failed or returned unusable data (a `RuntimeError` too,
    so existing `except RuntimeError` callers keep working)."""


class FatalError(RunError):
    """A failure that ends the run. `flag` is the `classify_run` keyword it sets."""

    flag = "simulator_failed"
    kind = "simulator"

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason)


class StartupFailure(FatalError):
    flag = "startup_failed"
    kind = "startup"


class OffboardFailure(FatalError):
    flag = "offboard_failed"
    kind = "offboard"


class EstimatorFailure(FatalError):
    flag = "estimator_failed"
    kind = "estimator"


class SimulatorFailure(FatalError):
    flag = "simulator_failed"
    kind = "simulator"


KIND_TO_ERROR = {
    "startup": StartupFailure,
    "offboard": OffboardFailure,
    "estimator": EstimatorFailure,
    "simulator": SimulatorFailure,
}


def is_recoverable(exc: BaseException) -> bool:
    return isinstance(exc, RecoverableError)


def failure_for_kind(kind: str, reason: str) -> FatalError:
    return KIND_TO_ERROR[kind](reason)


def outcome_flags(exc: BaseException) -> Dict[str, bool]:
    """The `classify_run` keyword flags for a fatal exception (unknown exceptions are
    infrastructure failures: they prove nothing about homing)."""
    flag = exc.flag if isinstance(exc, FatalError) else SimulatorFailure.flag
    return {flag: True}


def failure_reason(exc: BaseException) -> str:
    if isinstance(exc, FatalError):
        return exc.reason
    return "unexpected:{}: {}".format(type(exc).__name__, exc)


class RecoverableBudget(object):
    """Bounds recoverable errors: too many in a row, or too many overall, is fatal."""

    def __init__(
        self,
        max_consecutive: int = PROPOSED_RECOVERABLE_ERROR_MAX_CONSECUTIVE,
        max_total: int = PROPOSED_RECOVERABLE_ERROR_MAX_TOTAL,
    ) -> None:
        self.max_consecutive = int(max_consecutive)
        self.max_total = int(max_total)
        self.consecutive = 0
        self.total = 0
        self.last_error: Optional[BaseException] = None

    def ok(self) -> None:
        """A step succeeded: the consecutive count restarts (the total never does)."""
        self.consecutive = 0

    def note(self, exc: BaseException) -> Optional[FatalError]:
        """Count a recoverable error. Returns the fatal error it escalates to, or None."""
        self.consecutive += 1
        self.total += 1
        self.last_error = exc
        if self.consecutive > self.max_consecutive:
            return SimulatorFailure("recoverable_errors_consecutive:{}: {}".format(self.consecutive, exc))
        if self.total > self.max_total:
            return SimulatorFailure("recoverable_errors_total:{}: {}".format(self.total, exc))
        return None
