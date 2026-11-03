from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.frames import M_FLU_TO_FRD, flu_to_frd


def square_resolution() -> Tuple[int, int]:
    """Nearest square resolution to five megapixels."""
    return (2236, 2236)


def pinhole_intrinsics(width: int, height: int, fov_x_deg: float, fov_y_deg: float) -> dict:
    """Pinhole intrinsics with the principal point at the image center and zero distortion."""
    fx = (width / 2.0) / np.tan(np.deg2rad(fov_x_deg) / 2.0)
    fy = (height / 2.0) / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    return {
        "fx": fx,
        "fy": fy,
        "width": width,
        "height": height,
        "cx": width / 2.0,
        "cy": height / 2.0,
        "dist_coeffs": np.zeros(5),
    }


def boresight_frd(pitch_deg: float) -> np.ndarray:
    """Body-FRD optical axis. Positive pitch looks down, toward +z."""
    pitch = np.deg2rad(pitch_deg)
    return np.array([np.cos(pitch), 0.0, np.sin(pitch)])


def pitched_extrinsics(pitch_deg: float) -> dict:
    """
    Body-to-camera mount for a camera pitched down from the simulator-body forward axis.

    Camera +z is the simulator-body boresight (cos(pitch), 0, -sin(pitch)).
    Camera +x is body right (simulator body -y). Camera +y is body down
    (simulator body -z), tilted by the same pitch. orientation is [i, j, k, w].
    """
    pitch = np.deg2rad(pitch_deg)
    cosine = np.cos(pitch)
    sine = np.sin(pitch)
    # Rows are the camera axes written in the simulator body frame, so this
    # matrix is R_BC: body vectors map to camera vectors.
    body_to_camera = np.array([
        [0.0, -1.0, 0.0],
        [-sine, 0.0, -cosine],
        [cosine, 0.0, -sine],
    ])
    return {
        "position": np.zeros(3),
        "orientation": Rotation.from_matrix(body_to_camera).as_quat(),
    }


def pixel_to_camera_ray(u: float, v: float, intrinsics: dict) -> np.ndarray:
    """Unit camera-frame direction. The camera looks along +z, with +x right and +y down."""
    x = (u - intrinsics["cx"]) / intrinsics["fx"]
    y = (v - intrinsics["cy"]) / intrinsics["fy"]
    ray = np.array([x, y, 1.0], dtype=float)
    return ray / np.linalg.norm(ray)


@dataclass(frozen=True)
class CameraMount:
    """Camera optical centre and orientation in the body FRD frame.

    The optical frame has +x right in the image, +y down in the image and +z
    along the optical axis. `rotation_frd_from_camera` maps optical vectors to
    body FRD vectors.
    """

    position_frd_m: np.ndarray
    rotation_frd_from_camera: np.ndarray


# Gazebo sensor axes are x forward, y left, z up. Columns are the optical
# axes (right, down, forward) written in those sensor axes.
_SENSOR_FROM_OPTICAL = np.array([
    [0.0, 0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0, -1.0, 0.0],
])


def mount_from_gazebo_pose(position_flu_m, rpy_rad) -> CameraMount:
    """Camera mount from an sdf sensor pose in the body FLU frame.

    `rpy_rad` is the sdf (roll, pitch, yaw): R = Rz(yaw) Ry(pitch) Rx(roll).
    Positive pitch tilts the optical axis down.
    """
    roll, pitch, yaw = (float(angle) for angle in rpy_rad)
    sensor_in_flu = Rotation.from_euler("ZYX", [yaw, pitch, roll]).as_matrix()
    optical_in_flu = sensor_in_flu @ _SENSOR_FROM_OPTICAL
    return CameraMount(
        position_frd_m=flu_to_frd(np.asarray(position_flu_m, dtype=float)),
        rotation_frd_from_camera=M_FLU_TO_FRD @ optical_in_flu,
    )


def mount_from_pitch(position_flu_m, pitch_deg: float) -> CameraMount:
    """Mount whose optical axis is pitched `pitch_deg` below the body nose, positive down."""
    return mount_from_gazebo_pose(position_flu_m, (0.0, np.deg2rad(pitch_deg), 0.0))


def pixel_ray_ned(
    u: float,
    v: float,
    intrinsics: dict,
    mount: CameraMount,
    rotation_ned_from_frd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """NED ray of pixel (u, v): camera offset from the body origin and unit direction.

    Camera frame, then body FRD through the mount, then NED through the
    vehicle attitude. The offset is the rotated camera position, so the ray
    starts at the camera, not at the body origin.
    """
    camera_ray = pixel_to_camera_ray(u, v, intrinsics)
    attitude = np.asarray(rotation_ned_from_frd, dtype=float)
    direction = attitude @ (mount.rotation_frd_from_camera @ camera_ray)
    offset = attitude @ mount.position_frd_m
    return offset, direction


def project_ned_point(
    point_ned,
    body_position_ned,
    rotation_ned_from_frd: np.ndarray,
    intrinsics: dict,
    mount: CameraMount,
) -> Optional[Tuple[float, float]]:
    """Pixel (u, v) of a NED point, or None when it is not in front of the camera."""
    attitude = np.asarray(rotation_ned_from_frd, dtype=float)
    relative_frd = attitude.T @ (np.asarray(point_ned, dtype=float) - np.asarray(body_position_ned, dtype=float))
    in_camera = mount.rotation_frd_from_camera.T @ (relative_frd - mount.position_frd_m)
    if in_camera[2] <= 0.0:
        return None
    u = intrinsics["fx"] * in_camera[0] / in_camera[2] + intrinsics["cx"]
    v = intrinsics["fy"] * in_camera[1] / in_camera[2] + intrinsics["cy"]
    return float(u), float(v)
