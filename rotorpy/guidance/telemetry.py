"""Machine-readable flight telemetry: one JSONL file per run plus raw frames.

Every write is failure-tolerant. A recording error is counted and reported
through `last_error`; it never raises into the control loop, and a log that
keeps failing disables itself instead of retrying forever.
"""

from collections import deque
from dataclasses import dataclass, field, fields
from datetime import datetime, timezone
from enum import Enum
import json
import os
from pathlib import Path
import secrets
from typing import Any, Callable, Dict, List, Optional

import numpy as np

SCHEMA_VERSION = 1
TELEMETRY_FILE = "telemetry.jsonl"
FRAME_DIR = "frames"
_RESERVED_KINDS = ("metadata", "tick", "outcome")
_MAX_CONSECUTIVE_ERRORS = 5
# Proposed, not verified: frames kept either side of an acquisition, handoff or loss.
PROPOSED_EVENT_FRAMES_BEFORE = 5
PROPOSED_EVENT_FRAMES_AFTER = 5


def _jsonable(value: Any) -> Any:
    if isinstance(value, Enum):
        return _jsonable(value.value)
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def new_run_id() -> str:
    """UTC time plus a random suffix, so two runs in one second still differ."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return "{}-{}".format(stamp, secrets.token_hex(4))


def frame_to_float(frame: np.ndarray) -> np.ndarray:
    """uint8 RGB frame to the float32 [0, 1] image the guidance consumes."""
    return np.asarray(frame, dtype=np.float32) / 255.0


def decision_from_command(command: Any) -> Dict[str, Any]:
    """The perception and guidance decision carried by a `HomingCommand`."""
    return _jsonable({
        "velocity_ned": command.velocity_ned,
        "acceleration_ned": command.acceleration_ned,
        "yaw_rad": command.yaw_rad,
        "locked": command.locked,
        "tracking": command.tracking,
        "source": command.source,
        "tracker_center": command.tracker_center,
        "tracker_side": command.tracker_side,
        # `tracking.TrackingState`/reason (Gazebo and legacy paths alike); `None` from
        # any caller whose command predates this field (kept optional for that reason).
        "tracking_state": getattr(command, "tracking_state", None),
        "abort_reason": getattr(command, "abort_reason", None),
    })


def _candidate_entry(camera: str, obs: Any) -> Dict[str, Any]:
    side = None
    if obs.bounds is not None:
        side = max(obs.bounds[2] - obs.bounds[0], obs.bounds[3] - obs.bounds[1]) + 1
    return {
        "camera": camera,
        "status": obs.status.value,
        "center": None if obs.reference is None else list(obs.reference),
        "bounds": obs.bounds,
        "side": side,
        "area": obs.area,
        "confidence": obs.confidence,
        "clipped": obs.clipped,
        "num_candidates": obs.num_candidates,
        "timestamp": obs.timestamp,
    }


def detection_fields_from_command(command: Any) -> Dict[str, Any]:
    """`candidates`/`selected`/`confidence` `TickRecord` fields from a `HomingCommand`'s
    per-camera `pixels.TargetObservation`s (`command.detections`).

    `{}` when the command carries none (the legacy no-mounts path, or any caller that
    never set `detections`): the tick's `TickRecord` then keeps its unpopulated defaults
    instead of fabricating placeholder detection data.
    """
    detections = getattr(command, "detections", None)
    if not detections:
        return {}
    candidates = [
        _candidate_entry(camera, obs) for camera, obs in detections.items()
        if obs is not None and obs.status.value != "no_frame"
    ]
    selected = next((entry for entry in candidates if entry["camera"] == command.source), None)
    confidence = None if selected is None else selected["confidence"]
    return {"candidates": candidates, "selected": selected, "confidence": confidence}


def target_motion_fields_from_command(command: Any) -> Dict[str, Any]:
    """`target_motion` `TickRecord` field from a `HomingCommand`'s
    `target_estimator.TargetMotionEstimate` (`command.target_motion`).

    `{}` when the command carries none (the legacy no-mounts path, which never builds
    an estimator, or any caller that never set `target_motion`): the tick's
    `TickRecord` then keeps its unpopulated `None` default instead of fabricating
    placeholder motion data. `status != "observable"` means guidance has already
    fallen back to its existing bearing-only/pure-pursuit steering for this tick --
    the `reason` field records why.
    """
    estimate = getattr(command, "target_motion", None)
    if estimate is None:
        return {}
    return {"target_motion": _jsonable({
        "status": estimate.status,
        "reason": estimate.reason,
        "relative_position_ned": estimate.relative_position_ned,
        "relative_velocity_ned": estimate.relative_velocity_ned,
        "target_velocity_ned": estimate.target_velocity_ned,
        "closing_speed_m_s": estimate.closing_speed_m_s,
        "position_sigma_m": estimate.position_sigma_m,
        "velocity_sigma_m_s": estimate.velocity_sigma_m_s,
        "age_s": estimate.age_s,
    })}


def setpoint_fields_from_command(command: Any) -> Dict[str, Any]:
    """`requested_setpoint`/`limited_setpoint`/`limiting_reasons` `TickRecord` fields
    from a `HomingCommand` (plan Phase 7's vehicle-envelope/continuity limiting).

    `{}` when the command carries no `requested_velocity_ned` (the legacy no-mounts
    path, which has no envelope to limit against, or a mounts-path hold/terminal
    tick): the tick's `TickRecord` then keeps its unpopulated `None`/`[]` defaults,
    same convention as `target_motion_fields_from_command`. `limited_setpoint` is
    `command.velocity_ned` itself -- `command_limits.limit_command`'s output, already
    bounded to the vehicle envelope -- not a separate field; the caller (the Gazebo
    tracker) still reports the actually-transmitted `transmitted_setpoint` itself,
    since a startup-hold tick sends a different velocity than `command.velocity_ned`.
    """
    requested = getattr(command, "requested_velocity_ned", None)
    if requested is None:
        return {}
    return {
        "requested_setpoint": _jsonable(requested),
        "limited_setpoint": _jsonable(command.velocity_ned),
        "limiting_reasons": list(getattr(command, "limiting_reasons", None) or []),
    }


@dataclass
class RunMetadata:
    """What a run was: configuration, firmware, effective parameters, rendering, timing."""

    config: Dict[str, Any]
    firmware_id: str = "unknown"
    px4_params: Dict[str, Any] = field(default_factory=dict)
    rendering_mode: str = "unknown"
    sim_timing: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self, run_id: str) -> Dict[str, Any]:
        record = _jsonable({
            "config": self.config,
            "firmware_id": self.firmware_id,
            "px4_params": self.px4_params,
            "rendering_mode": self.rendering_mode,
            "sim_timing": self.sim_timing,
        })
        record.update(
            schema=SCHEMA_VERSION,
            kind="metadata",
            run_id=run_id,
            created_utc=datetime.now(timezone.utc).isoformat(),
        )
        return record


@dataclass
class TickRecord:
    """One control tick. Vectors are NED / FRD lists; times are seconds."""

    seq: int = 0
    t_s: Optional[float] = None
    image_capture_s: Dict[str, Optional[float]] = field(default_factory=dict)
    image_receipt_s: Dict[str, Optional[float]] = field(default_factory=dict)
    processing_start_s: Optional[float] = None
    processing_end_s: Optional[float] = None
    attitude_quat: Optional[List[float]] = None
    position_ned: Optional[List[float]] = None
    velocity_ned: Optional[List[float]] = None
    attitude_age_s: Optional[float] = None
    position_age_s: Optional[float] = None
    velocity_age_s: Optional[float] = None
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    camera: Optional[str] = None
    target_motion: Optional[Dict[str, Any]] = None
    requested_setpoint: Optional[List[float]] = None
    limited_setpoint: Optional[List[float]] = None
    transmitted_setpoint: Optional[List[float]] = None
    limiting_reasons: List[str] = field(default_factory=list)
    measured_velocity_ned: Optional[List[float]] = None
    state: Optional[str] = None
    px4_mode: Optional[str] = None
    failure_reason: Optional[str] = None
    # A snapshot of `SetpointStream.stats` (publish_count/interval/observation-age/
    # expired_count/failed) at this tick, so a dead setpoint stream is visible in
    # telemetry instead of only inferable from the absence of commanded motion.
    setpoint_stats: Optional[Dict[str, Any]] = None
    # Inputs of the guidance call, present only on ticks that ran it.
    guidance_update: Optional[Dict[str, Any]] = None
    decision: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        record = _jsonable({item.name: getattr(self, item.name) for item in fields(self)})
        record.update(schema=SCHEMA_VERSION, kind="tick")
        return record

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TickRecord":
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in data.items() if key in known})


class TelemetryLog(object):
    """Append-only JSONL log of one run. Never overwrites an existing run."""

    def __init__(self, run_id: Optional[str], run_dir: Optional[Path], handle) -> None:
        self.run_id = run_id
        self.run_dir = run_dir
        self.path = None if run_dir is None else run_dir / TELEMETRY_FILE
        self._file = handle
        self.enabled = handle is not None
        self.last_error: Optional[BaseException] = None
        self._consecutive_errors = 0

    @classmethod
    def create(
        cls,
        root,
        metadata: RunMetadata,
        id_factory: Callable[[], str] = new_run_id,
        attempts: int = 10,
    ) -> "TelemetryLog":
        """Open `root/<run_id>/telemetry.jsonl`. Raises if no free run id is found."""
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        for _ in range(int(attempts)):
            run_id = id_factory()
            run_dir = root / run_id
            try:
                run_dir.mkdir()
            except FileExistsError:
                continue
            handle = open(run_dir / TELEMETRY_FILE, "x", encoding="utf-8", buffering=1)
            log = cls(run_id, run_dir, handle)
            handle.write(json.dumps(metadata.to_dict(run_id)) + "\n")
            handle.flush()
            return log
        raise FileExistsError("no unused run id under {}".format(root))

    @classmethod
    def start_tolerant(cls, root, metadata: RunMetadata, **kwargs) -> "TelemetryLog":
        """Like `create`, but a failure gives a disabled log instead of raising."""
        try:
            return cls.create(root, metadata, **kwargs)
        except Exception as exc:
            log = cls(None, None, None)
            log.last_error = exc
            return log

    def write_tick(self, tick: TickRecord) -> bool:
        return self._write(tick.to_dict)

    def write_event(self, kind: str, **fields_) -> bool:
        def record():
            if kind in _RESERVED_KINDS:
                raise ValueError("reserved event kind {}".format(kind))
            return _jsonable(dict(fields_, schema=SCHEMA_VERSION, kind=kind))

        return self._write(record)

    def finish(self, outcome, reason: Optional[str] = None, **fields_) -> bool:
        def record():
            return _jsonable(dict(
                fields_, schema=SCHEMA_VERSION, kind="outcome",
                outcome=outcome.value if isinstance(outcome, Enum) else str(outcome),
                reason=reason,
            ))

        return self._write(record)

    def close(self) -> None:
        if self._file is None:
            return
        try:
            self._file.close()
        except Exception as exc:
            self.last_error = exc
        self.enabled = False

    def _write(self, build: Callable[[], Dict[str, Any]]) -> bool:
        if not self.enabled:
            return False
        try:
            self._file.write(json.dumps(build()) + "\n")
            self._file.flush()
        except Exception as exc:
            self.last_error = exc
            self._consecutive_errors += 1
            if self._consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                self.enabled = False
            return False
        self._consecutive_errors = 0
        return True


@dataclass
class TelemetryRun:
    """A parsed run. `truncated_lines` counts unreadable lines, e.g. a crashed tail."""

    path: Path
    metadata: Dict[str, Any]
    ticks: List[TickRecord]
    events: List[Dict[str, Any]]
    outcome: Optional[Dict[str, Any]]
    truncated_lines: int

    @property
    def run_dir(self) -> Path:
        return self.path.parent


def read_telemetry(path) -> TelemetryRun:
    """Read a telemetry file, or the file inside a run directory."""
    path = Path(path)
    if path.is_dir():
        path = path / TELEMETRY_FILE
    metadata: Dict[str, Any] = {}
    ticks: List[TickRecord] = []
    events: List[Dict[str, Any]] = []
    outcome = None
    truncated = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                kind = record["kind"]
            except (ValueError, KeyError, TypeError):
                truncated += 1
                continue
            if kind == "metadata":
                metadata = record
            elif kind == "tick":
                ticks.append(TickRecord.from_dict(record))
            elif kind == "outcome":
                outcome = record
            else:
                events.append(record)
    return TelemetryRun(path, metadata, ticks, events, outcome, truncated)


class FrameStore(object):
    """Lossless uint8 `.npy` frames under `<run_dir>/frames/`.

    Frame names are a pure function of tick and camera, so a tick record can
    name its frames before they are saved. By default only the frames around
    an event are kept: the last `pre` ticks are held in memory and written
    when `mark_event` is called, then the next `post` ticks follow.
    `save_all` keeps every frame, which full replay needs.
    """

    def __init__(self, run_dir, pre: int = 0, post: int = 0, save_all: bool = False) -> None:
        self.run_dir = Path(run_dir)
        self.pre = int(pre)
        self.post = int(post)
        self.save_all = bool(save_all)
        self.last_error: Optional[BaseException] = None
        self._ring: deque = deque()
        self._latest_seq = -1
        self._post_left = 0
        self._saving_seq: Optional[int] = None

    @staticmethod
    def name(seq: int, camera: str) -> str:
        return "{}/{:06d}_{}.npy".format(FRAME_DIR, int(seq), camera)

    def exists(self, name: str) -> bool:
        return (self.run_dir / name).is_file()

    def load(self, name: str) -> np.ndarray:
        return np.load(self.run_dir / name, allow_pickle=False)

    def offer(self, seq: int, camera: str, frame: np.ndarray) -> Optional[str]:
        """Save or hold one frame. Returns its name if it was written."""
        try:
            if seq != self._latest_seq:
                self._latest_seq = int(seq)
                if self._post_left > 0:
                    self._post_left -= 1
                    self._saving_seq = int(seq)
                else:
                    self._saving_seq = None
            if self.save_all or self._saving_seq == seq:
                return self._save(seq, camera, frame)
            self._ring.append((int(seq), camera, frame))
            while self._ring and self._ring[0][0] <= self._latest_seq - self.pre:
                self._ring.popleft()
        except Exception as exc:
            self.last_error = exc
        return None

    def mark_event(self) -> None:
        """Write the held frames and keep the next `post` ticks."""
        try:
            held, self._ring = list(self._ring), deque()
            for seq, camera, frame in held:
                self._save(seq, camera, frame)
            self._post_left = self.post
        except Exception as exc:
            self.last_error = exc

    def _save(self, seq: int, camera: str, frame: np.ndarray) -> str:
        try:
            os.mkdir(str(self.run_dir / FRAME_DIR))
        except FileExistsError:
            pass
        name = self.name(seq, camera)
        np.save(str(self.run_dir / name), np.asarray(frame, dtype=np.uint8), allow_pickle=False)
        return name


class EventDetector(object):
    """Names acquisition, handoff, loss and reacquisition from steering state."""

    def __init__(self) -> None:
        self._tracking = False
        self._source: Optional[str] = None
        self._ever_tracked = False

    def update(self, tracking: bool, source: Optional[str]) -> Optional[str]:
        event = None
        if tracking and not self._tracking:
            event = "reacquisition" if self._ever_tracked else "acquisition"
        elif tracking and source != self._source:
            event = "handoff"
        elif not tracking and self._tracking:
            event = "loss"
        self._tracking = bool(tracking)
        self._source = source if tracking else None
        self._ever_tracked = self._ever_tracked or bool(tracking)
        return event


class RunRecorder(object):
    """A telemetry log, its frame store and event detection behind one tolerant call."""

    def __init__(self, log: TelemetryLog, frames: Optional[FrameStore] = None) -> None:
        self.log = log
        self.frames = frames
        self._events = EventDetector()

    @property
    def enabled(self) -> bool:
        return self.log.enabled

    def step(
        self,
        seq: int,
        t_s: float,
        height_m: float,
        attitude_quat,
        velocity_ned,
        forward: Optional[np.ndarray],
        upward: Optional[np.ndarray],
        updated: bool,
        command: Any,
        **tick_fields,
    ) -> bool:
        """Record one tick. `forward`/`upward` are the uint8 frames given to guidance.

        Returns False, never raises, when anything about recording fails.
        """
        try:
            update = None
            if updated:
                update = {
                    "t": t_s,
                    "height_m": height_m,
                    "attitude_quat": attitude_quat,
                    "velocity_ned": velocity_ned,
                    "forward_frame": self._offer(seq, "forward", forward),
                    "upward_frame": self._offer(seq, "upward", upward),
                }
            event = self._events.update(command.tracking, command.source)
            if event is not None:
                self.log.write_event(event, seq=seq, t_s=t_s)
                if self.frames is not None:
                    self.frames.mark_event()
            tick = TickRecord(
                seq=seq,
                t_s=t_s,
                attitude_quat=attitude_quat,
                velocity_ned=velocity_ned,
                guidance_update=update,
                decision=decision_from_command(command),
                **tick_fields
            )
            return self.log.write_tick(tick)
        except Exception as exc:
            self.log.last_error = exc
            return False

    def event(self, kind: str, **fields_) -> bool:
        try:
            return self.log.write_event(kind, **fields_)
        except Exception as exc:
            self.log.last_error = exc
            return False

    def finish(self, outcome, reason: Optional[str] = None, **fields_) -> bool:
        try:
            return self.log.finish(outcome, reason=reason, **fields_)
        except Exception as exc:
            self.log.last_error = exc
            return False

    def _offer(self, seq: int, camera: str, frame: Optional[np.ndarray]) -> Optional[str]:
        """Name of the frame a replay must find, or None when no picture was given."""
        if frame is None:
            return None
        if self.frames is not None:
            self.frames.offer(seq, camera, frame)
        return FrameStore.name(seq, camera)


def environment_metadata(config: Dict[str, Any], environ=None) -> RunMetadata:
    """Run metadata from the launch environment. PX4 parameters are not queried here."""
    environ = os.environ if environ is None else environ
    return RunMetadata(
        config=config,
        firmware_id=environ.get("PX4_FIRMWARE_ID", "unknown"),
        px4_params={},
        rendering_mode="software" if environ.get("LIBGL_ALWAYS_SOFTWARE") == "1" else "gpu",
        sim_timing={
            "camera_rate_hz": config.get("camera_rate_hz"),
            "timeout_s": config.get("timeout_s"),
        },
    )


def start_run_telemetry(root, config: Dict[str, Any], environ=None) -> RunRecorder:
    """A recorder for a new run under `root`. Never raises; disabled if setup fails."""
    environ = os.environ if environ is None else environ
    log = TelemetryLog.start_tolerant(root, environment_metadata(config, environ))
    frames = None
    if log.enabled:
        frames = FrameStore(
            log.run_dir,
            pre=PROPOSED_EVENT_FRAMES_BEFORE,
            post=PROPOSED_EVENT_FRAMES_AFTER,
            save_all=environ.get("HOMING_SAVE_ALL_FRAMES") == "1",
        )
    return RunRecorder(log, frames)
