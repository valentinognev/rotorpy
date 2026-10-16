from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


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
