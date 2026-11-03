"""Independent periodic MAVLink setpoint publishing, decoupled from perception timing.

`SetpointStream` holds the latest guidance command and publishes it on its
own cadence, unrelated to the camera/guidance update rate. It never blocks on
perception: `set_command` just stores the newest value and the wall time of
the observation it came from. `tick` decides, independent of when
`set_command` last ran, whether to publish the held command, the configured
failure response, or nothing -- once the held command's age exceeds
`expiry_s`, or after `mark_fatal()`, the normal command stops regardless of
heartbeat/armed/offboard state (that state lives elsewhere and is irrelevant
here on purpose: a stale command must not stay valid just because PX4 last
reported armed+offboard).

`set_command`/`mark_fatal` run on the caller's step/control thread; `tick`
runs on the independent periodic-publish thread. A lock protects the
(command, command_wall_s) pair and the fatal/stats state so a concurrent
`tick` never observes a command paired with a *different* call's timestamp.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

# Proposed, not verified: a commanded setpoint older than this is no longer trustworthy.
PROPOSED_SETPOINT_EXPIRY_S = 0.5

# Proposed, not verified: the documented default failure response once a command expires
# or `mark_fatal()` is called -- send nothing at all. This deliberately lets PX4 lose its
# offboard setpoint stream and fall back to its own configured failsafe (RTL/land/hold)
# rather than this tracker fabricating a "safe" velocity command of its own; callers that
# want a different response (for example an explicit hold command) pass their own
# `failure_command` instead of relying on this default silently.
PROPOSED_FAILURE_COMMAND: Optional[Any] = None


@dataclass
class SetpointStats:
    """Running stats `SetpointStream` exposes for monitoring and tests.

    `publish_count`/`last_interval_s`/`mean_interval_s`/`max_interval_s` count only ticks
    that actually called `publish` (a real command or failure-response send), not every
    `tick()` call -- a tick that sends nothing (no command ever set, default failure
    response) must not count as a healthy publish.
    """

    publish_count: int = 0
    last_interval_s: Optional[float] = None
    mean_interval_s: Optional[float] = None
    max_interval_s: Optional[float] = None
    last_observation_age_s: Optional[float] = None
    max_observation_age_s: Optional[float] = None
    expired_count: int = 0
    failed: bool = False


class SetpointStream(object):
    def __init__(
        self,
        publish: Callable[[Any], None],
        expiry_s: float = PROPOSED_SETPOINT_EXPIRY_S,
        failure_command: Optional[Any] = PROPOSED_FAILURE_COMMAND,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._publish = publish
        self._expiry_s = float(expiry_s)
        self._failure_command = failure_command
        self._now = now
        self._lock = threading.Lock()
        self._command: Optional[Any] = None
        self._command_wall_s: Optional[float] = None
        self._last_publish_s: Optional[float] = None
        self._fatal = False
        self.stats = SetpointStats()

    def set_command(self, command: Any, observation_wall_s: float) -> None:
        """Record the newest guidance command and the wall time of the observation it used."""
        with self._lock:
            self._command = command
            self._command_wall_s = float(observation_wall_s)

    def peek(self) -> Tuple[Optional[Any], Optional[float]]:
        """The held `(command, command_wall_s)` pair, read atomically."""
        with self._lock:
            return self._command, self._command_wall_s

    def stats_snapshot(self) -> Dict[str, Any]:
        """A consistent `dict` snapshot of `stats`, safe to read from another thread
        (for example to surface in telemetry) while `tick()` is concurrently updating it."""
        with self._lock:
            return dict(vars(self.stats))

    def mark_fatal(self) -> None:
        """A fatal failure stops the normal command even with a fresh observation."""
        with self._lock:
            self._fatal = True
            self.stats.failed = True

    def tick(self, now_s: Optional[float] = None) -> Optional[Any]:
        """Publish and return the right thing for this tick: command, failure response, or None."""
        now_s = float(self._now() if now_s is None else now_s)
        with self._lock:
            expired = self._command_wall_s is None or (now_s - self._command_wall_s) > self._expiry_s
            if expired and self._command_wall_s is not None:
                self.stats.expired_count += 1
            to_send = self._failure_command if (self._fatal or expired) else self._command
            if self._command_wall_s is not None:
                age = now_s - self._command_wall_s
                self.stats.last_observation_age_s = age
                self.stats.max_observation_age_s = (
                    age if self.stats.max_observation_age_s is None else max(self.stats.max_observation_age_s, age)
                )
            if to_send is not None:
                if self._last_publish_s is not None:
                    interval = now_s - self._last_publish_s
                    self.stats.last_interval_s = interval
                    self.stats.max_interval_s = (
                        interval if self.stats.max_interval_s is None else max(self.stats.max_interval_s, interval)
                    )
                    n = self.stats.publish_count
                    previous_mean = 0.0 if self.stats.mean_interval_s is None else self.stats.mean_interval_s
                    self.stats.mean_interval_s = (previous_mean * (n - 1) + interval) / n if n else interval
                self._last_publish_s = now_s
                self.stats.publish_count += 1
        # Publish outside the lock: I/O must not block `set_command`/`mark_fatal`, and a
        # callback that re-enters this stream must not deadlock on a non-reentrant lock.
        if to_send is not None:
            self._publish(to_send)
        return to_send
