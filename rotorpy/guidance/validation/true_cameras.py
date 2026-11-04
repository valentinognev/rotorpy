"""True-NED/FRD synthetic cameras for the offline reliability harness (Task 10 fix).

Root cause (see `task-10-report.md`'s investigation log): `homing_config.
gazebo_scene_from_config` builds its `PinholeCamera`s via `scene.nominal_camera`/
`camera_geometry.pitched_extrinsics`, which assumes the *standard* (non-mirrored)
"simulator body" convention documented in `frames.py` (body +Y = left, matching
`frames.frd_to_sim_body`) -- correct for that function's actual callers (the legacy
Python-only example's own camera, and `gazebo_record.py`'s replay-only preview
rendering, neither of which is ever cross-checked pixel-for-pixel against the
Gazebo-mounts path's own backprojection). But `HomingGuidance`'s Gazebo-mounts
backprojection (`homing._sightline_and_relative`, via `camera_geometry.
mount_from_gazebo_pose`/`pixel_ray_ned` and `gazebo_bridge.attitude_homing_to_ned_frd`)
expects the vehicle attitude it is given (`attitude_sim_quat`, as this harness's own
`vehicle_model.attitude_from_specific_force` builds it) to be in the *homing
simulator* convention (body +Y = right, matching `gazebo_bridge.attitude_gazebo_to_homing`).
These two "simulator body" conventions differ by a left-right mirror. Composing
`gazebo_scene_from_config`'s camera (expects the standard convention) with
`attitude_sim_quat` (actually the mirrored one) is self-consistent only when the
vehicle's yaw exactly matches its initial pursuit heading (the one case every prior
unit test exercised) and diverges, pixel-for-pixel, as the vehicle yaws away from it
-- confirmed empirically against `camera_geometry.project_ned_point` (the same
projection `homing.py`'s backprojection is built on): the rendered pixel and the
backprojected pixel are an exact mirror image of each other around the image's
horizontal center, growing with yaw deviation and vanishing only at the one
boresight-aligned heading. This is what fed the `TargetMotionEstimator` a spurious,
steadily growing "target velocity" for a genuinely stationary target as the pursuing
vehicle yawed, eventually crashing it (see the report for the full trace).

Fix: render entirely in the SAME true NED (world) / FRD (body) convention the
Gazebo-mounts backprojection already uses, sidestepping the "simulator body"
mirroring question entirely -- built directly from the same `homing_config.
gazebo_camera_mounts` the real `HomingGuidance` uses, so consistency holds by
construction, not by two independently-derived conventions coincidentally agreeing.
Harness-only: not imported by `rotorpy.guidance.homing`/`tracking`/`target_estimator`/
`command_limits`, and never fed anything but the vehicle's own simulated state.
"""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.frames import FrameConvention
from rotorpy.guidance.camera_geometry import CameraMount, pinhole_intrinsics
from rotorpy.guidance.homing_config import gazebo_camera_mounts
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.world import World


def _camera_from_mount(mount: CameraMount, intrinsics: dict, **kwargs) -> PinholeCamera:
    """A `PinholeCamera` whose extrinsics describe `mount` directly in true body-FRD
    (not the mismatched "simulator body" convention `scene.pitched_extrinsics` uses).

    `mount.rotation_frd_from_camera` maps camera vectors to FRD; `PinholeCamera` wants
    the inverse (FRD, i.e. body, to camera) as its `orientation` extrinsic. Both are
    proper rotations, so the inverse is just the transpose.
    """
    rotation_body_from_camera = mount.rotation_frd_from_camera.T
    return PinholeCamera(
        intrinsics=intrinsics,
        extrinsics={
            "position": np.array(mount.position_frd_m, dtype=float, copy=True),
            "orientation": Rotation.from_matrix(rotation_body_from_camera).as_quat(),
        },
        **kwargs,
    )


def build_true_frd_cameras(config: dict) -> Tuple[PinholeCamera, PinholeCamera]:
    """Forward and upward `PinholeCamera`s, in true NED world / FRD body, from `config`.

    Mirrors `homing_config._camera_kwargs`'s rendering parameters (resolution, FOV,
    frame rate, splat radius, feature output) so this stays a drop-in replacement for
    `gazebo_scene_from_config`'s cameras everywhere except extrinsics convention.
    """
    intrinsics = pinhole_intrinsics(
        int(config["image_width_px"]), int(config["image_height_px"]),
        float(config["fov_x_deg"]), float(config["fov_y_deg"]),
    )
    camera_kwargs = dict(
        frame_rate=config["camera_rate_hz"],
        splat_radius=config["splat_radius_px"],
        feature_output=config["feature_output"],
    )
    forward_mount, upward_mount = gazebo_camera_mounts(config)
    forward_camera = _camera_from_mount(forward_mount, intrinsics, **camera_kwargs)
    upward_camera = _camera_from_mount(upward_mount, intrinsics, **camera_kwargs)
    return forward_camera, upward_camera


def world_to_true_ned(world: World) -> World:
    """`world` (as `target_world.MovingTargetWorldBuilder` builds it, in the
    "simulator" north/east/up convention) re-expressed in true NED (down, not up).

    A pure z-negation of every block extent and feature point -- the same relation
    `frames.ned_to_sim`/`sim_to_ned` use elsewhere, applied here to geometry instead
    of a single vector. Colors/descriptors carry no spatial meaning and pass through
    unchanged. `blocks`' (zmin, zmax) swap along with the sign flip so zmin <= zmax
    still holds.
    """
    data = world.world
    blocks = []
    for block in data.get("blocks", []):
        xmin, xmax, ymin, ymax, zmin, zmax = (float(value) for value in block["extents"])
        blocks.append({"extents": [xmin, xmax, ymin, ymax, -zmax, -zmin], "color": block.get("color")})
    features_config = data.get("features")
    features = None
    if features_config is not None and "points" in features_config:
        points = np.asarray(features_config["points"], dtype=float).copy()
        if points.size:
            points[:, 2] *= -1.0
        features = dict(features_config, points=points)
    world_data = {"bounds": {"extents": data["bounds"]["extents"]}, "blocks": blocks}
    if features is not None:
        world_data["features"] = features
    return World(world_data, frame=FrameConvention.FRD_NED)
