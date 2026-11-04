"""Cross-checks that keep the homing scene defined in one place.

`examples/camera_homing.jsonc` is the authoritative scene. The Gazebo camera
model, the target model and the launch script each repeat some of its values,
and `check_scene_consistency` reports every disagreement as one message. Run it
from the test suite, and before a flight with `check_repo_scene_consistency`.

Validating analytic projections against live Gazebo images (protocol, not run
offline). It needs a flight environment this repository's tests do not have.
1. Spawn a small, saturated red marker at a known world position, on the
   forward camera's optical axis and again about 400 px off axis.
2. Record the camera image, `/gazebo/get_model_state` for the vehicle and the
   `CameraInfo` message. Check `validate_camera_info` on that message.
3. Detect the marker centroid in the saved lossless frame, and project the known
   marker position with `rotorpy.guidance.camera_geometry.project_ned_point`
   using the recorded pose, the configured intrinsics and the mount.
4. Repeat for the upward camera with the vehicle pitched steeply down.
5. Accept only if every centroid is within about 2 px of its prediction.
   Gazebo's `CameraInfo` principal point is `(size + 1) / 2`, one pixel from
   this module's `size / 2`; a larger or lateral-sign error means the mount or
   frame convention is wrong.
"""

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List

import numpy as np

from rotorpy.guidance.camera_geometry import mount_from_gazebo_pose, mount_from_pitch, pinhole_intrinsics

# Proposed, not verified: Gazebo's CameraInfo puts the principal point one pixel
# beyond ours, and the lens is built from a rounded field of view.
PROPOSED_PRINCIPAL_POINT_TOLERANCE_PX = 1.5
PROPOSED_FOCAL_RELATIVE_TOLERANCE = 1e-3

_ANGLE_TOLERANCE_RAD = 1e-6
_LENGTH_TOLERANCE_M = 1e-6


def config_intrinsics(config: dict) -> dict:
    """Pinhole intrinsics of the configured lens."""
    return pinhole_intrinsics(
        int(config["image_width_px"]),
        int(config["image_height_px"]),
        float(config["fov_x_deg"]),
        float(config["fov_y_deg"]),
    )


def camera_info_dict(message) -> dict:
    """A ROS `sensor_msgs/CameraInfo` as the plain dict `validate_camera_info` reads."""
    return {
        "width": int(message.width),
        "height": int(message.height),
        "distortion_model": str(message.distortion_model),
        "D": [float(value) for value in message.D],
        "K": [float(value) for value in message.K],
        "P": [float(value) for value in message.P],
    }


def validate_camera_info(
    config: dict,
    camera_info: dict,
    pixel_tolerance: float = PROPOSED_PRINCIPAL_POINT_TOLERANCE_PX,
    relative_tolerance: float = PROPOSED_FOCAL_RELATIVE_TOLERANCE,
) -> List[str]:
    """Differences between the configured intrinsics and a `CameraInfo`-like dict.

    The dict carries `width`, `height`, `K` (row-major 3x3), and optionally `D`
    and `P` (row-major 3x4). An empty list means they agree.
    """
    problems: List[str] = []
    try:
        width, height = int(camera_info["width"]), int(camera_info["height"])
        matrix = [float(value) for value in camera_info["K"]]
    except (KeyError, TypeError, ValueError) as error:
        return ["camera info needs width, height and K: {}".format(error)]
    if len(matrix) != 9 or not all(math.isfinite(value) for value in matrix):
        return ["camera info K must hold nine finite numbers"]
    expected = config_intrinsics(config)
    if width != expected["width"]:
        problems.append("width {} differs from the configured {}".format(width, expected["width"]))
    if height != expected["height"]:
        problems.append("height {} differs from the configured {}".format(height, expected["height"]))

    def compare(name, measured, wanted, tolerance, relative):
        limit = tolerance * abs(wanted) if relative else tolerance
        if not abs(measured - wanted) <= limit:
            problems.append("{} {:.6g} differs from the configured {:.6g}".format(name, measured, wanted))

    compare("fx", matrix[0], expected["fx"], relative_tolerance, True)
    compare("fy", matrix[4], expected["fy"], relative_tolerance, True)
    compare("cx", matrix[2], expected["cx"], pixel_tolerance, False)
    compare("cy", matrix[5], expected["cy"], pixel_tolerance, False)
    if abs(matrix[1]) > 1e-9 or abs(matrix[3]) > 1e-9:
        problems.append("K has skew; the configured lens has none")
    coefficients = camera_info.get("D")
    if coefficients is not None and any(abs(float(value)) > 1e-9 for value in coefficients):
        problems.append("distortion coefficients are not zero; the configured lens is distortion free")
    projection = camera_info.get("P")
    if projection is not None:
        projection = [float(value) for value in projection]
        if len(projection) != 12 or not all(math.isfinite(value) for value in projection):
            problems.append("camera info P must hold twelve finite numbers")
        else:
            for name, index, value in (
                ("fx", 0, matrix[0]), ("cx", 2, matrix[2]), ("fy", 5, matrix[4]), ("cy", 6, matrix[5]),
            ):
                if abs(projection[index] - value) > max(1e-6 * abs(value), 1e-9):
                    problems.append("P[{}] ({}) disagrees with K".format(index, name))
    return problems


def _floats(text: str) -> List[float]:
    return [float(value) for value in text.split()]


def parse_camera_sdf(text: str) -> Dict[str, dict]:
    """Camera sensors of the vehicle model, keyed `forward` and `upward`."""
    root = ET.fromstring(text)
    cameras = {}
    for sensor in root.iter("sensor"):
        if sensor.get("type") != "camera":
            continue
        pose = _floats(sensor.findtext("pose"))
        camera = sensor.find("camera")
        cameras[sensor.get("name").replace("_camera", "")] = {
            "position_flu_m": pose[:3],
            "rpy_rad": pose[3:],
            "horizontal_fov_rad": float(camera.findtext("horizontal_fov")),
            "width": int(camera.findtext("image/width")),
            "height": int(camera.findtext("image/height")),
            "update_rate_hz": float(sensor.findtext("update_rate")),
        }
    return cameras


def parse_cylinder_sdf(text: str) -> Dict[str, float]:
    """Visual and collision cylinder size of the target model."""
    root = ET.fromstring(text)
    sizes = {}
    for kind, prefix in (("visual", "visual_"), ("collision", "")):
        cylinder = root.find(".//{}/geometry/cylinder".format(kind))
        sizes[prefix + "radius_m"] = float(cylinder.findtext("radius"))
        sizes[prefix + "length_m"] = float(cylinder.findtext("length"))
    return sizes


_NUMBER = r"(-?[0-9]+(?:\.[0-9]+)?)"


def parse_launch_spawns(text: str) -> dict:
    """Spawn poses in the launch script. Positions are Gazebo east, north, up."""

    def pose(model: str, trailer: str = ""):
        pattern = r"--model-name=\"?{}\"?\s+-x\s+{}\s+-y\s+{}\s+-z\s+{}{}".format(
            model, _NUMBER, _NUMBER, _NUMBER, trailer
        )
        found = re.search(pattern, text)
        return None if found is None else tuple(float(value) for value in found.groups())

    yaw = re.search(r"math\.atan2\({}\s*,\s*{}\)".format(_NUMBER, _NUMBER), text)
    return {
        "target_enu": pose("homing_target"),
        "vehicle_enu": pose("hummingbird_0", r"\s+-Y"),
        "vehicle_yaw_enu_rad": None if yaw is None else math.atan2(float(yaw.group(1)), float(yaw.group(2))),
    }


def parse_takeoff_spawn(text: str):
    """The ground-spawn pose used only when `HOMING_STARTUP_MODE=takeoff` -- a second
    `hummingbird_0` spawn line in the same script (fix round 1, minor e), distinct from
    the one `parse_launch_spawns` finds (the first, airborne-spawn one). Anchored on the
    script's own "(takeoff mode)" marker so it is found regardless of which branch
    `parse_launch_spawns`'s plain first-match search happens to pick up."""
    pattern = r"\(takeoff mode\)[\s\S]*?--model-name=\"?hummingbird_0\"?\s+-x\s+{0}\s+-y\s+{0}\s+-z\s+{0}".format(
        _NUMBER
    )
    found = re.search(pattern, text)
    return None if found is None else tuple(float(value) for value in found.groups())


def _angle_difference(first: float, second: float) -> float:
    return abs(math.atan2(math.sin(first - second), math.cos(first - second)))


def _check_cameras(config: dict, vehicle_sdf: str, problems: List[str]) -> None:
    cameras = parse_camera_sdf(vehicle_sdf)
    expected = config_intrinsics(config)
    for name in ("forward", "upward"):
        if name not in cameras:
            problems.append("{} camera is missing from the vehicle model".format(name))
            continue
        camera = cameras[name]
        wanted_position = np.asarray(config["{}_camera_position_flu_m".format(name)], dtype=float)
        if not np.allclose(camera["position_flu_m"], wanted_position, atol=_LENGTH_TOLERANCE_M):
            problems.append("{} camera position {} in the model differs from the configured {}".format(
                name, camera["position_flu_m"], wanted_position.tolist()))
        roll, pitch, yaw = camera["rpy_rad"]
        pitch_deg = float(config["{}_pitch_deg".format(name)])
        if _angle_difference(pitch, math.radians(pitch_deg)) > _ANGLE_TOLERANCE_RAD * 10:
            problems.append("{} camera pitch {:.6f} rad in the model differs from the configured {:.6f}".format(
                name, pitch, math.radians(pitch_deg)))
        if abs(roll) > _ANGLE_TOLERANCE_RAD or abs(yaw) > _ANGLE_TOLERANCE_RAD:
            problems.append("{} camera has roll or yaw in the model; the configuration has none".format(name))
        built = mount_from_gazebo_pose(camera["position_flu_m"], camera["rpy_rad"])
        wanted = mount_from_pitch(wanted_position, pitch_deg)
        if not (np.allclose(built.position_frd_m, wanted.position_frd_m, atol=_LENGTH_TOLERANCE_M)
                and np.allclose(built.rotation_frd_from_camera, wanted.rotation_frd_from_camera, atol=1e-5)):
            problems.append("{} camera mount differs between the model and the configuration".format(name))
        if camera["width"] != expected["width"] or camera["height"] != expected["height"]:
            problems.append("{} camera image size {}x{} differs from the configured {}x{}".format(
                name, camera["width"], camera["height"], expected["width"], expected["height"]))
        focal = (camera["width"] / 2.0) / math.tan(camera["horizontal_fov_rad"] / 2.0)
        if abs(focal - expected["fx"]) > PROPOSED_FOCAL_RELATIVE_TOLERANCE * expected["fx"]:
            problems.append("{} camera fov: model focal length {:.2f} px, configured fov_x_deg gives {:.2f} px".format(
                name, focal, expected["fx"]))
        if abs(focal - expected["fy"]) > PROPOSED_FOCAL_RELATIVE_TOLERANCE * expected["fy"]:
            problems.append("{} camera fov: model has square pixels, configured fov_y_deg gives fy {:.2f} px".format(
                name, expected["fy"]))
        if abs(camera["update_rate_hz"] - float(config["camera_rate_hz"])) > 1e-9:
            problems.append("{} camera update_rate {} Hz differs from camera_rate_hz {}".format(
                name, camera["update_rate_hz"], config["camera_rate_hz"]))


def _check_target_and_launch(config: dict, target_sdf: str, launch_script: str, problems: List[str]) -> None:
    cylinder = parse_cylinder_sdf(target_sdf)
    radius, length = float(config["gazebo_target_radius_m"]), float(config["gazebo_target_height_m"])
    for label, prefix in (("visual", "visual_"), ("collision", "")):
        if abs(cylinder[prefix + "radius_m"] - radius) > _LENGTH_TOLERANCE_M:
            problems.append("target {} radius {} differs from gazebo_target_radius_m {}".format(
                label, cylinder[prefix + "radius_m"], radius))
        if abs(cylinder[prefix + "length_m"] - length) > _LENGTH_TOLERANCE_M:
            problems.append("target {} cylinder length {} differs from gazebo_target_height_m {}".format(
                label, cylinder[prefix + "length_m"], length))
    spawns = parse_launch_spawns(launch_script)
    target_ned = np.asarray(config["target_ground_ned"], dtype=float)
    # A cylinder model is positioned by its centre, half its length above the ground contact.
    wanted_target = (float(target_ned[1]), float(target_ned[0]), -float(target_ned[2]) + length / 2.0)
    if spawns["target_enu"] is None or not np.allclose(spawns["target_enu"], wanted_target, atol=_LENGTH_TOLERANCE_M):
        problems.append("target spawn {} differs from the configured cylinder centre {}".format(
            spawns["target_enu"], wanted_target))
    start_ned = np.asarray(config["start_ned"], dtype=float)
    wanted_vehicle = (float(start_ned[1]), float(start_ned[0]), -float(start_ned[2]))
    if spawns["vehicle_enu"] is None or not np.allclose(
        spawns["vehicle_enu"], wanted_vehicle, atol=_LENGTH_TOLERANCE_M
    ):
        problems.append("vehicle spawn {} differs from the configured start {}".format(
            spawns["vehicle_enu"], wanted_vehicle))
    # The spawn heading aims the nose along the start-to-target course. This is
    # scene setup only; the guidance never reads it.
    wanted_yaw = math.atan2(target_ned[0] - start_ned[0], target_ned[1] - start_ned[1])
    yaw = spawns["vehicle_yaw_enu_rad"]
    if yaw is None or _angle_difference(yaw, wanted_yaw) > _ANGLE_TOLERANCE_RAD * 10:
        problems.append("vehicle spawn yaw {} rad differs from the configured course heading {:.6f}".format(
            yaw, wanted_yaw))
    # Fix round 1 (minor e): the takeoff-mode ground spawn ("-x 0 -y 0 -z 0.2 -Y ...",
    # only taken when HOMING_STARTUP_MODE=takeoff) was never checked -- the airborne
    # spawn above is the only hummingbird_0 line `parse_launch_spawns` ever sees (the
    # first match in the script). A deferred import avoids a circular import: this
    # module is itself imported by `homing_config`.
    from rotorpy.guidance.homing_config import PROPOSED_TAKEOFF_SPAWN_HEIGHT_M

    takeoff = parse_takeoff_spawn(launch_script)
    wanted_takeoff = (wanted_vehicle[0], wanted_vehicle[1], PROPOSED_TAKEOFF_SPAWN_HEIGHT_M)
    if takeoff is None or not np.allclose(takeoff, wanted_takeoff, atol=_LENGTH_TOLERANCE_M):
        problems.append("takeoff-mode ground spawn {} differs from the expected {} (same north/east, "
                         "PROPOSED_TAKEOFF_SPAWN_HEIGHT_M above the ground)".format(takeoff, wanted_takeoff))


def check_scene_consistency(config: dict, vehicle_sdf: str, target_sdf: str, launch_script: str) -> List[str]:
    """Every disagreement between the config and the model, target and launch texts."""
    problems: List[str] = []
    _check_cameras(config, vehicle_sdf, problems)
    _check_target_and_launch(config, target_sdf, launch_script, problems)
    return problems


def check_repo_scene_consistency(config: dict, root) -> List[str]:
    """`check_scene_consistency` on this repository's own files under `root`."""
    root = Path(root)
    return check_scene_consistency(
        config,
        (root / "gazeboModels" / "hummingbird" / "hummingbird.sdf.jinja").read_text(),
        (root / "gazeboModels" / "homing" / "target_cylinder.sdf").read_text(),
        (root / "Dockerfiles" / "sitl_hummingbird_run.sh").read_text(),
    )
