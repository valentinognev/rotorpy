"""Re-run perception and guidance on recorded inputs, without Gazebo."""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from rotorpy.guidance.homing_config import guidance_from_config
from rotorpy.guidance.telemetry import (
    FrameStore,
    _jsonable,
    decision_from_command,
    frame_to_float,
    read_telemetry,
)


@dataclass
class ReplayResult:
    """`decisions` holds {seq, decision} per replayed tick, in order.

    `mismatches` are the ticks whose replayed decision differs from the
    recorded one. `gap_seq` is the first tick whose frames were not saved;
    guidance is stateful, so replay stops there. `skipped` counts ticks that
    ran no guidance (stale command reuse).
    """

    decisions: List[Dict[str, Any]] = field(default_factory=list)
    mismatches: List[int] = field(default_factory=list)
    replayed: int = 0
    skipped: int = 0
    gap_seq: Optional[int] = None

    @property
    def matches(self) -> bool:
        return not self.mismatches


def replay_run(path, guidance_factory: Callable[[dict], Any] = guidance_from_config) -> ReplayResult:
    """Feed a recorded run's guidance inputs to a fresh guidance and compare decisions."""
    run = read_telemetry(path)
    guidance = guidance_factory(run.metadata["config"])
    frames = FrameStore(run.run_dir)
    result = ReplayResult()
    for tick in run.ticks:
        inputs = tick.guidance_update
        if inputs is None:
            result.skipped += 1
            continue
        images = {}
        for camera in ("forward", "upward"):
            name = inputs.get("{}_frame".format(camera))
            if name is not None and not frames.exists(name):
                result.gap_seq = tick.seq
                return result
            images[camera] = None if name is None else frame_to_float(frames.load(name))
        command = guidance.update(
            inputs["t"],
            inputs["height_m"],
            np.array(inputs["attitude_quat"], dtype=float),
            np.array(inputs["velocity_ned"], dtype=float),
            images["forward"],
            images["upward"],
        )
        decision = _jsonable(decision_from_command(command))
        result.decisions.append({"seq": tick.seq, "decision": decision})
        result.replayed += 1
        if decision != tick.decision:
            result.mismatches.append(tick.seq)
    return result
