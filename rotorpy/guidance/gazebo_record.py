"""Record a Gazebo homing flight into the same page as the RotorPy example."""

from pathlib import Path
from typing import Optional

import numpy as np

from rotorpy.guidance.frames import ned_to_sim
from rotorpy.guidance.gazebo_bridge import gazebo_position_to_sim
from rotorpy.guidance.homing_config import gazebo_scene_from_config
from rotorpy.guidance.homing_page import write_homing_page
from rotorpy.guidance.runner import (
    HomingRecord,
    ReplayGeometry,
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


def _camera_geometry(config, name):
    return {
        "position_flu_m": [float(value) for value in config["{}_camera_position_flu_m".format(name)]],
        "pitch_deg": float(config["{}_pitch_deg".format(name)]),
        "fov_x_deg": float(config["fov_x_deg"]),
        "fov_y_deg": float(config["fov_y_deg"]),
        "width_px": int(config["image_width_px"]),
        "height_px": int(config["image_height_px"]),
    }


def replay_geometry(config, target_times=None, target_ground_sim=None) -> ReplayGeometry:
    """The target and cameras a replay shows, from the authoritative `config`.

    `target_*` is the sampled ground contact of a moving target (simulator frame); omit
    them to show the configured static target.
    """
    configured = ned_to_sim(np.asarray(config["target_ground_ned"], dtype=float))
    path = None if not target_ground_sim else np.vstack(target_ground_sim)
    return ReplayGeometry(
        target_radius_m=float(config["gazebo_target_radius_m"]),
        target_height_m=float(config["gazebo_target_height_m"]),
        target_ground_sim=configured if path is None else path[0],
        target_path_time=None if path is None else np.asarray(target_times, dtype=float),
        target_path_sim=path,
        cameras={"forward": _camera_geometry(config, "forward"), "upward": _camera_geometry(config, "upward")},
    )


class GazeboHomingRecorder(object):
    """Poses and both camera previews for one Gazebo homing flight."""

    def __init__(self, config):
        self._config = config
        self.target_times = []
        self.target_positions = []
        self.world, self.forward_camera, self.upward_camera = gazebo_scene_from_config(config)
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

    def observe(self, stamp, position_enu, quaternion_homing, forward, upward, command, target_ground_ned=None):
        """Store one pose (and the target's ground contact when sampled). Store both
        pictures only when both are present.

        Returns the legacy stop reason, `ground` or `timeout`, or None. The tracker's
        termination does not depend on this recorder; it uses `scoring.TruthScorer`.
        """
        elapsed = self._elapsed(stamp)
        if target_ground_ned is not None:
            self.target_times.append(elapsed)
            self.target_positions.append(ned_to_sim(np.asarray(target_ground_ned, dtype=float)))
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
            geometry=replay_geometry(self._config, self.target_times, self.target_positions),
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
