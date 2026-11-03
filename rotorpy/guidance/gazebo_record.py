"""Record a Gazebo homing flight into the same page as the RotorPy example."""

from pathlib import Path
from typing import Optional

import numpy as np

from rotorpy.guidance.gazebo_bridge import gazebo_position_to_sim
from rotorpy.guidance.homing_config import scene_from_config
from rotorpy.guidance.homing_page import write_homing_page
from rotorpy.guidance.runner import (
    HomingRecord,
    _tracker_sample,
    flight_status,
    preview_image,
    preview_step,
)

# The Gazebo body rests a few centimetres above z = 0.
_GAZEBO_GROUND_Z = 0.25


def gazebo_page_path(config_path) -> Path:
    """`Results/camera_homing_gazebo.html` next to the examples directory."""
    root = Path(config_path).resolve().parents[1]
    return root / "Results" / "camera_homing_gazebo.html"


class GazeboHomingRecorder(object):
    """Poses and both camera previews for one Gazebo homing flight."""

    def __init__(self, config):
        _state, self.world, self.forward_camera, self.upward_camera = scene_from_config(config)
        self.preview = preview_step(
            int(config["image_height_px"]),
            int(config["image_width_px"]),
        )
        self.timeout_s = float(config["timeout_s"])
        self._t0 = None
        self._written = False
        self.times = []
        self.positions = []
        self.quaternions = []
        self.winds = []
        self.image_times = []
        self.forward_images = []
        self.upward_images = []
        self.sources = []
        self.tracker_centers = []
        self.tracker_sides = []

    def observe(self, stamp, position_enu, quaternion_homing, forward, upward, command):
        """Store one pose. Store both pictures only when both are present.

        Returns the stop reason, `ground` or `timeout`, or None.
        """
        elapsed = self._elapsed(stamp)
        position = gazebo_position_to_sim(position_enu)
        self.times.append(elapsed)
        self.positions.append(position)
        self.quaternions.append(np.array(quaternion_homing, dtype=float, copy=True))
        self.winds.append(np.zeros(3))
        if forward is not None and upward is not None:
            source, center, side = _tracker_sample(command)
            self.image_times.append(elapsed)
            self.forward_images.append(preview_image(forward))
            self.upward_images.append(preview_image(upward))
            self.sources.append(source)
            self.tracker_centers.append(center)
            self.tracker_sides.append(side)
        return flight_status(elapsed, float(position[2]), self.timeout_s, _GAZEBO_GROUND_Z)

    def record(self) -> Optional[HomingRecord]:
        if not self.image_times:
            return None
        return HomingRecord(
            time=np.asarray(self.times, dtype=float),
            position=np.vstack(self.positions),
            quaternion=np.vstack(self.quaternions),
            wind=np.vstack(self.winds),
            image_time=np.asarray(self.image_times, dtype=float),
            forward_image=np.stack(self.forward_images),
            upward_image=np.stack(self.upward_images),
            source=list(self.sources),
            tracker_center=np.vstack(self.tracker_centers),
            tracker_side=np.asarray(self.tracker_sides, dtype=float),
            world=self.world,
            forward_camera=self.forward_camera,
            upward_camera=self.upward_camera,
            preview_step=self.preview,
        )

    def write(self, path) -> Optional[Path]:
        """Write the page once. Later calls return the same path."""
        destination = Path(path)
        if self._written:
            return destination
        recorded = self.record()
        if recorded is None:
            return None
        written = write_homing_page(recorded, destination)
        self._written = True
        return written

    def _elapsed(self, stamp):
        stamp = float(stamp)
        if self._t0 is None:
            self._t0 = stamp
        return stamp - self._t0
