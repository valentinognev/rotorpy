"""Pure, rospy-free helper for `CameraHomingTracker.step()`'s command/stamp decision.

`SetpointStream.set_command(command, observation_wall_s)` restamps the held command's
wall-clock age: a stale restamp hides a dead camera or dead PX4 estimate forever (the
expiry never fires, so a dead link never stops the stream). `command_and_stamp` is the
decision that avoids that, extracted so it is unit-testable without rospy or a live
MAVLink connection: it decides what `step()` should hold as the current command, and
separately, the wall time (or "do not restamp") that the setpoint stream should be told
about.
"""

from typing import Any, Callable, Optional, Tuple


def command_and_stamp(
    have_estimate: bool,
    fresh: bool,
    previous_command: Optional[Any],
    placeholder_command: Any,
    compute_command: Callable[[], Any],
    observation_wall_s: float,
) -> Tuple[Any, Optional[float]]:
    """Decide this tick's command and the setpoint-stream stamp.

    Returns `(command, stamp_wall_s)`. `stamp_wall_s` is `None` exactly when the command
    is unchanged from `previous_command` (the hold-last-command branch): the setpoint
    stream's clock must not be refreshed just because `step()` ran again, only when a
    genuinely new command was produced -- from a fresh image, or (once, before the first
    image) from the first PX4 estimate establishing the initial course. That is what lets
    a dead camera or dead PX4 estimate make the held command age past `expiry_s` and stop
    the normal command, instead of looking perpetually fresh forever.

    `compute_command` is called (not pre-computed by the caller) only when a new command
    is actually needed, so a caller that only wants the decision does not have to run
    guidance unconditionally.
    """
    if have_estimate and (fresh or previous_command is None):
        return compute_command(), observation_wall_s
    if previous_command is not None:
        return previous_command, None
    return placeholder_command, observation_wall_s
