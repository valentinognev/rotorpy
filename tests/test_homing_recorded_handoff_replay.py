"""Replay-style regression using the real frames from `Results/camera_homing_gazebo.html`.

The plan's documented baseline-failure evidence (`docs/superpowers/plans/
2026-11-03-gazebo-homing-reliability.md`, "Evidence and baseline failures"): a single
recorded flight showed an upward-camera transition at approximately 2.14 s, then no
further steering past approximately 2.30 s despite later forward-camera target
visibility. These exact recorded frames (not synthesized ones) are decoded here and fed
through `HomingGuidance`, with the old `_upward_latched`/`_permanent_loss` behavior
replaced by the recoverable tracking state machine (plan Phase 5).

The recording's own attitude/velocity/height are not stored in the page (it only keeps
rendering-derived 3D state, not the raw PX4 pose or the pixel-level detector's actual
run); this test uses a fixed, plausible nose-down attitude instead (chosen so the
decoded forward-camera pixels project to a downward-looking ray; see the module-level
`_ATTITUDE` comment). It is not a claim that this reproduces the original flight's exact
commanded velocities -- only that it reproduces the same recorded *pixels* seen by the
same two cameras in the same order, which is what the state machine acts on.
"""

import base64
import io
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import pinhole_intrinsics
from rotorpy.guidance.homing import HomingGuidance
from rotorpy.guidance.homing_config import gazebo_camera_mounts, gazebo_target_reference, load_homing_config
from rotorpy.guidance.tracking import TrackingState

_HTML_PATH = Path(__file__).resolve().parents[1] / "Results" / "camera_homing_gazebo.html"
_CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"
_PREVIEW_SIZE = 279  # The html's embedded preview resolution, not the full 2236 px frame.
# A fixed nose-down pitch so the recorded forward-camera pixels (which sit high in the
# frame once decoded) project to a downward-looking ray at every replayed tick: the
# recording keeps no raw attitude to replay exactly (see module docstring).
_ATTITUDE = Rotation.from_euler("y", 35.0, degrees=True).as_quat()
_VELOCITY_NED = np.array([10.0, 0.0, 0.0])
_HEIGHT_M = 30.0


def _load_snapshots():
    content = _HTML_PATH.read_text()
    match = re.search(r'<script type="application/json" id="homing-data">(.*?)</script>', content, re.S)
    payload = json.loads(match.group(1))
    return payload["times"], payload["snapshots"]


def _decode(data_url):
    raw = base64.b64decode(data_url.split(",", 1)[1])
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image, dtype=float) / 255.0


def _guidance():
    config = load_homing_config(_CONFIG_PATH)
    forward_mount, upward_mount = gazebo_camera_mounts(config)
    intrinsics = pinhole_intrinsics(
        _PREVIEW_SIZE, _PREVIEW_SIZE, float(config["fov_x_deg"]), float(config["fov_y_deg"])
    )
    reference = gazebo_target_reference(config)
    return HomingGuidance(
        nav_constant=float(config["nav_constant"]),
        lock_pixels=int(config["lock_pixels"]),
        initial_velocity_ned=_VELOCITY_NED,
        intrinsics=intrinsics,
        forward_extrinsics=None,
        upward_extrinsics=None,
        forward_mount=forward_mount,
        upward_mount=upward_mount,
        target_reference_height_m=reference.reference_height_m,
    )


def test_recorded_early_handoff_no_longer_permanently_disables_guidance():
    times, snapshots = _load_snapshots()
    guidance = _guidance()
    sources = []
    states = []
    # Indices 7-19 of the recording: forward-only steering, the recorded upward
    # handoff at t ~= 2.14 s (index 9), the recorded loss at t ~= 2.30 s (index 10),
    # and several further ticks where the forward camera is recorded as still visible.
    for index in range(7, 20):
        forward_image = _decode(snapshots[index]["forward_image"])
        upward_image = _decode(snapshots[index]["upward_image"])
        command = guidance.update(times[index], _HEIGHT_M, _ATTITUDE, _VELOCITY_NED, forward_image, upward_image)
        sources.append(command.source)
        states.append(command.tracking_state)
        assert command.tracking_state is not TrackingState.ABORT

    assert sources[0] == "forward"  # index 7, t ~= 1.64 s: forward-only, as recorded.
    # Index 9 (t ~= 2.14 s): the recorded upward-camera transition.
    assert sources[2] == "upward"
    # Index 10 onward (t ~= 2.30 s on): the recorded loss. Unlike the original flight
    # (and the old permanent-loss code), guidance recovers through the still-visible
    # forward camera instead of going silent for the rest of the run.
    assert all(source == "forward" for source in sources[3:])
    assert all(state is TrackingState.TRACK for state in states[3:])


def test_forward_camera_is_independently_detectable_in_every_later_recorded_frame():
    """Evidence, not steering: the plan's claim that the forward camera "kept seeing
    the target" past the recorded loss, checked directly against `detect_target` on
    the raw recorded pixels, independent of `HomingGuidance`'s own internal state."""
    from rotorpy.guidance.pixels import DetectionStatus, detect_target

    times, snapshots = _load_snapshots()
    for index in range(10, 20):
        forward_image = _decode(snapshots[index]["forward_image"])
        obs = detect_target(forward_image, timestamp=times[index], camera="forward")
        assert obs.status is DetectionStatus.DETECTED
