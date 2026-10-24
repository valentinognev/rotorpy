"""Nominal Hummingbird camera-homing scene: start state, world, and camera."""

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import pinhole_intrinsics, pitched_extrinsics, square_resolution
from rotorpy.guidance.frames import ned_to_sim
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.world import World

START_NED = np.array([0.0, 0.0, -30.0])
TARGET_GROUND_NED = np.array([100.0, 50.0, 0.0])
INITIAL_SPEED = 10.0
FEATURE_SPACING = 0.05
SPLAT_RADIUS = 1


def initial_velocity_ned(start_ned: np.ndarray, target_ground_ned: np.ndarray, speed: float) -> np.ndarray:
    """Horizontal NED velocity of magnitude `speed` from the start xy toward the target xy."""
    start = np.asarray(start_ned, dtype=float)
    target = np.asarray(target_ground_ned, dtype=float)
    horizontal = target[:2] - start[:2]
    direction = horizontal / np.linalg.norm(horizontal)
    return np.array([direction[0] * speed, direction[1] * speed, 0.0])


def hover_rotor_speed(quad_params: dict) -> float:
    """Rotor speed, rad/s, such that four rotors together support the vehicle weight."""
    return float(np.sqrt(quad_params["mass"] * 9.81 / (4.0 * quad_params["k_eta"])))


def nominal_initial_state(
    start_ned=None,
    target_ground_ned=None,
    initial_speed=None,
    body_rate=None,
    wind=None,
) -> dict:
    """Level Hummingbird state at the start, aimed along the horizontal course.

    Omitted arguments use the nominal start, target, and 10 m/s course.
    Rotor speeds are the hover speed from the Hummingbird weight. Yaw is the
    heading of that course, which matches atan2 of the target east and north
    while the start is on the north-east origin.
    """
    start = START_NED if start_ned is None else np.asarray(start_ned, dtype=float)
    target = TARGET_GROUND_NED if target_ground_ned is None else np.asarray(target_ground_ned, dtype=float)
    speed = INITIAL_SPEED if initial_speed is None else float(initial_speed)
    velocity_ned = initial_velocity_ned(start, target, speed)
    yaw = float(np.arctan2(velocity_ned[1], velocity_ned[0]))
    omega = hover_rotor_speed(quad_params)
    return {
        "x": ned_to_sim(start),
        "v": ned_to_sim(velocity_ned),
        "q": Rotation.from_euler("z", yaw).as_quat(),
        "w": np.zeros(3) if body_rate is None else np.asarray(body_rate, dtype=float),
        "wind": np.zeros(3) if wind is None else np.asarray(wind, dtype=float),
        "rotor_speeds": np.full(4, omega),
    }


def nominal_world(
    bounds=None,
    extents=None,
    color=None,
    feature_spacing=None,
    feature_mode="regular",
    descriptor_noise=0,
) -> World:
    """One red 0.5 m by 0.5 m by 1 m block on the target ground point."""
    world_data = {
        "bounds": {"extents": [-50, 300, -50, 200, -20, 80] if bounds is None else list(bounds)},
        "blocks": [{
            "extents": [99.75, 100.25, 49.75, 50.25, 0.0, 1.0] if extents is None else list(extents),
            "color": [1, 0, 0] if color is None else list(color),
        }],
    }
    return World(
        world_data,
        add_features=True,
        feature_mode=feature_mode,
        feature_spacing=FEATURE_SPACING if feature_spacing is None else float(feature_spacing),
        descriptor_noise=descriptor_noise,
    )


def _nominal_intrinsics(width=None, height=None, fov_x_deg=90.0, fov_y_deg=90.0) -> dict:
    """Square pinhole lens. Distortion is a tuple so two cameras compare with ==."""
    default_width, default_height = square_resolution()
    if width is None:
        width = default_width
    if height is None:
        height = default_height
    intrinsics = pinhole_intrinsics(int(width), int(height), float(fov_x_deg), float(fov_y_deg))
    intrinsics["dist_coeffs"] = tuple(float(coeff) for coeff in intrinsics["dist_coeffs"])
    return intrinsics


def _mounted_extrinsics(pitch_deg, position):
    """pitched_extrinsics, with the body position replaced when one is given."""
    extrinsics = pitched_extrinsics(pitch_deg)
    if position is None:
        return extrinsics
    return {
        "position": np.array(position, dtype=float, copy=True),
        "orientation": np.array(extrinsics["orientation"], dtype=float, copy=True),
    }


def nominal_camera(
    pitch_deg=10,
    width=None,
    height=None,
    fov_x_deg=90.0,
    fov_y_deg=90.0,
    position=None,
    frame_rate=20,
    splat_radius=None,
    feature_output="rgb",
) -> PinholeCamera:
    """2236 by 2236 pinhole camera, 90 by 90 degree FOV, pitched 10 degrees down."""
    return PinholeCamera(
        intrinsics=_nominal_intrinsics(width, height, fov_x_deg, fov_y_deg),
        extrinsics=_mounted_extrinsics(pitch_deg, position),
        frame_rate=frame_rate,
        splat_radius=SPLAT_RADIUS if splat_radius is None else int(splat_radius),
        feature_output=feature_output,
    )


def nominal_upward_camera(
    pitch_deg=-70,
    width=None,
    height=None,
    fov_x_deg=90.0,
    fov_y_deg=90.0,
    position=None,
    frame_rate=20,
    splat_radius=None,
    feature_output="rgb",
) -> PinholeCamera:
    """2236 by 2236 pinhole camera, 90 by 90 degree FOV, pitched 70 degrees up."""
    return PinholeCamera(
        intrinsics=_nominal_intrinsics(width, height, fov_x_deg, fov_y_deg),
        extrinsics=_mounted_extrinsics(pitch_deg, position),
        frame_rate=frame_rate,
        splat_radius=SPLAT_RADIUS if splat_radius is None else int(splat_radius),
        feature_output=feature_output,
    )
