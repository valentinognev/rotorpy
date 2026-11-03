"""Bounded latest-observation buffering for camera images, independent per camera.

Each camera gets its own depth-1 "latest wins" slot: a new image always
replaces whatever is there, and a fresh image from one camera never marks
the other camera's cached image as fresh (the previous tracker shared one
`fresh` flag across both cameras, which let a stale picture from the camera
that did not just update ride along as if it were new). An observation that
was never read before being overwritten increments a drop counter so lost
work is visible instead of silently accumulating latency.
"""

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

# Channel count and BGR-vs-RGB byte order for each supported ROS-style encoding.
# Encodings not listed here are rejected explicitly instead of being decoded as garbage.
_ENCODINGS = {
    "rgb8": (3, False),
    "bgr8": (3, True),
    "rgba8": (4, False),
    "bgra8": (4, True),
    "mono8": (1, False),
}


def decode_image(data: bytes, width: int, height: int, encoding: str, step: int) -> np.ndarray:
    """Decode a raw row-major image buffer into a uint8 RGB array, shape (height, width, 3).

    `step` is the row stride in bytes (`sensor_msgs/Image.step`); it may exceed
    `width * channels` when rows are padded, and the padding is skipped. Alpha
    channels are dropped. `mono8` is replicated across three equal channels
    (a true grayscale camera cannot see the homing target's red color; this
    only makes a mono8 source decodable, not color-capable). Unsupported
    encodings and buffers too small for the declared stride/height raise
    `ValueError` instead of reshaping garbage.
    """
    if encoding not in _ENCODINGS:
        raise ValueError(
            "unsupported image encoding {!r}; supported encodings are {}".format(
                encoding, sorted(_ENCODINGS)
            )
        )
    channels, is_bgr_order = _ENCODINGS[encoding]
    width = int(width)
    height = int(height)
    step = int(step)
    row_bytes = width * channels
    if step < row_bytes:
        raise ValueError(
            "row stride {} is smaller than one tight row ({} bytes for {}x{} {})".format(
                step, row_bytes, width, channels, encoding
            )
        )
    buffer = np.frombuffer(data, dtype=np.uint8)
    needed = step * height
    if buffer.size < needed:
        raise ValueError(
            "image buffer has {} bytes, needs {} for height {} at stride {}".format(
                buffer.size, needed, height, step
            )
        )
    rows = buffer[:needed].reshape(height, step)[:, :row_bytes]
    frame = rows.reshape(height, width, channels)
    if is_bgr_order:
        order = [2, 1, 0, 3] if channels == 4 else [2, 1, 0]
        frame = frame[:, :, order]
    if channels == 4:
        frame = frame[:, :, :3]
    elif channels == 1:
        frame = np.repeat(frame, 3, axis=2)
    return np.array(frame, dtype=np.uint8)


@dataclass
class CameraObservation:
    """One decoded camera frame. `capture_s` is the image header stamp (may be unknown)."""

    capture_s: Optional[float]
    receipt_s: float
    image: np.ndarray
    encoding: str


class CameraSlot(object):
    """Depth-1 latest-wins slot for one camera.

    A new offer always replaces whatever observation is held, whether or not
    it was read. `dropped` counts observations that were overwritten before
    ever being taken as fresh, i.e. obsolete work that was discarded instead
    of queued.
    """

    def __init__(self) -> None:
        self._observation: Optional[CameraObservation] = None
        self._fresh = False
        self.dropped = 0

    def offer(self, capture_s: Optional[float], receipt_s: float, image: np.ndarray, encoding: str) -> None:
        if self._fresh:
            self.dropped += 1
        self._observation = CameraObservation(
            capture_s=None if capture_s is None else float(capture_s),
            receipt_s=float(receipt_s),
            image=image,
            encoding=encoding,
        )
        self._fresh = True

    def take_fresh(self) -> Tuple[Optional[CameraObservation], bool]:
        """The latest observation, and whether it is being read fresh for the first time."""
        was_fresh = self._fresh
        self._fresh = False
        return self._observation, was_fresh


class ObservationBuffer(object):
    """Independent latest-observation slots, one per named camera."""

    def __init__(self, cameras=("forward", "upward")) -> None:
        self._slots: Dict[str, CameraSlot] = {name: CameraSlot() for name in cameras}

    def offer(
        self,
        camera: str,
        capture_s: Optional[float],
        receipt_s: float,
        raw_bytes: bytes,
        width: int,
        height: int,
        encoding: str,
        step: int,
    ) -> None:
        image = decode_image(raw_bytes, width=width, height=height, encoding=encoding, step=step)
        self._slots[camera].offer(capture_s, receipt_s, image, encoding)

    def take_fresh(self, camera: str) -> Tuple[Optional[CameraObservation], bool]:
        return self._slots[camera].take_fresh()

    def dropped(self, camera: str) -> int:
        return self._slots[camera].dropped
