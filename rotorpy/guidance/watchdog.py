"""Wall-time staleness watchdogs for comms links: heartbeat, telemetry, image.

Each `Watchdog` tracks only wall-clock time on purpose, not simulation time:
a comms link that stops delivering data is a real-world failure that must be
caught even if the simulation itself is paused. `CommsWatchdogs` bundles
several named watchdogs (for example heartbeat, telemetry and image) so the
tracker can check all of them in one place.
"""

from typing import Dict, Optional


class Watchdog(object):
    """Flags staleness once wall-clock age since the last `touch` exceeds `timeout_s`.

    Before the first `touch`, the link is stale (there is nothing to be fresh
    about) and `age` is `None`.
    """

    def __init__(self, timeout_s: float) -> None:
        self._timeout_s = float(timeout_s)
        self._last_s: Optional[float] = None

    def touch(self, now_s: float) -> None:
        self._last_s = float(now_s)

    def age(self, now_s: float) -> Optional[float]:
        return None if self._last_s is None else float(now_s) - self._last_s

    def stale(self, now_s: float) -> bool:
        age = self.age(now_s)
        return age is None or age > self._timeout_s


class CommsWatchdogs(object):
    """A named bundle of `Watchdog`s, one per comms link."""

    def __init__(self, timeouts_s: Dict[str, float]) -> None:
        self._watchdogs: Dict[str, Watchdog] = {name: Watchdog(timeout) for name, timeout in timeouts_s.items()}

    def touch(self, name: str, now_s: float) -> None:
        self._watchdogs[name].touch(now_s)

    def stale(self, name: str, now_s: float) -> bool:
        return self._watchdogs[name].stale(now_s)

    def any_stale(self, now_s: float) -> Dict[str, bool]:
        """The subset of links that are currently stale, mapped to `True`."""
        return {name: True for name, watchdog in self._watchdogs.items() if watchdog.stale(now_s)}
