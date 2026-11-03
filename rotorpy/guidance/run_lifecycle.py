"""One unambiguous outcome per run, and an orderly flush/close on every termination path.

`RunLifecycle` is the single place a run's outcome is decided and its sinks are closed:

* `finish` records the outcome once. The first outcome wins; later calls are refused, so
  a run can never end with two outcome records. The outcome is also *announced* (one
  `RUN_OUTCOME {json}` line on stdout) independent of any recorder, so a run without
  telemetry, or with a failing disk, still ends with exactly one visible outcome.
* `ensure_outcome` is the shutdown net: a run that is being torn down with no outcome
  gets an infrastructure `SIMULATOR_FAILURE`, never silence.
* `close` runs every closer (recorder flush/close, frame flush, ...), isolating failures
  so one failing sink cannot stop the rest or raise into shutdown.

Nothing here depends on the recorder, the camera, or Gazebo truth.
"""

import json
from typing import Callable, Iterable, List, Optional, Tuple

from rotorpy.guidance.outcome import RunOutcome

OUTCOME_PREFIX = "RUN_OUTCOME"

OutcomeSink = Callable[[RunOutcome, Optional[str], dict], bool]


def _print_line(line: str) -> None:
    print(line, flush=True)


class RunLifecycle(object):
    def __init__(
        self,
        write_outcome: Optional[OutcomeSink],
        closers: Iterable[Tuple[str, Callable[[], None]]] = (),
        announce: Callable[[str], None] = _print_line,
    ) -> None:
        self._write = write_outcome
        self._closers: List[Tuple[str, Callable[[], None]]] = list(closers)
        self._announce = announce
        self.outcome: Optional[RunOutcome] = None
        self.outcome_reason: Optional[str] = None
        self.outcome_persisted = False
        self._fields: dict = {}
        self._closed = False
        self.errors: List[Tuple[str, BaseException]] = []

    def add_closer(self, name: str, closer: Callable[[], None]) -> None:
        self._closers.append((name, closer))

    @property
    def finished(self) -> bool:
        return self.outcome is not None

    def _persist(self) -> bool:
        if self._write is None:
            return False
        try:
            return bool(self._write(self.outcome, self.outcome_reason, dict(self._fields)))
        except Exception as exc:
            self.errors.append(("outcome", exc))
            return False

    def finish(self, outcome: RunOutcome, reason: Optional[str] = None, **fields) -> bool:
        """Decide and record the run's outcome. True only if this call recorded it."""
        if self.outcome is not None:
            return False
        self.outcome = RunOutcome(outcome)
        self.outcome_reason = reason
        self._fields = dict(fields)
        self.outcome_persisted = self._persist()
        try:
            self._announce("{} {}".format(OUTCOME_PREFIX, json.dumps(
                dict(fields, outcome=self.outcome.value, reason=reason, persisted=self.outcome_persisted),
                default=str,
            )))
        except Exception as exc:
            self.errors.append(("announce", exc))
        return self.outcome_persisted

    def ensure_outcome(self, reason: str) -> None:
        """Teardown with no outcome yet: record an infrastructure failure instead of silence."""
        if self.outcome is None:
            self.finish(RunOutcome.SIMULATOR_FAILURE, reason)

    def close(self) -> List[Tuple[str, BaseException]]:
        """Run every closer once; returns the `(name, exception)` failures it saw."""
        if self._closed:
            return []
        self._closed = True
        failures: List[Tuple[str, BaseException]] = []
        if self.outcome is not None and not self.outcome_persisted:
            # One last attempt to persist a decided-but-unwritten outcome.
            self.outcome_persisted = self._persist()
        for name, closer in self._closers:
            try:
                closer()
            except Exception as exc:
                failures.append((name, exc))
        self.errors.extend(failures)
        return failures

    def terminate(self, outcome: RunOutcome, reason: Optional[str] = None, **fields) -> List[Tuple[str, BaseException]]:
        """`finish` then `close`: the orderly end of a run."""
        self.finish(outcome, reason, **fields)
        return self.close()
