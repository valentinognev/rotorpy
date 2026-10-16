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


def nominal_initial_state() -> dict:
    """Level Hummingbird state at the nominal start, aimed at the target ground point."""
    velocity_ned = initial_velocity_ned(START_NED, TARGET_GROUND_NED, INITIAL_SPEED)
    yaw = float(np.arctan2(TARGET_GROUND_NED[1], TARGET_GROUND_NED[0]))
    omega = hover_rotor_speed(quad_params)
    return {
        "x": ned_to_sim(START_NED),
        "v": ned_to_sim(velocity_ned),
        "q": Rotation.from_euler("z", yaw).as_quat(),
        "w": np.zeros(3),
        "wind": np.zeros(3),
        "rotor_speeds": np.full(4, omega),
    }


def nominal_world() -> World:
    """One red 0.5 m by 0.5 m by 1 m block on the target ground point."""
    world_data = {
        "bounds": {"extents": [-50, 300, -50, 200, -20, 80]},
        "blocks": [{
            "extents": [99.75, 100.25, 49.75, 50.25, 0.0, 1.0],
            "color": [1, 0, 0],
        }],
    }
    return World(
        world_data,
        add_features=True,
        feature_mode="regular",
        feature_spacing=FEATURE_SPACING,
        descriptor_noise=0,
    )


def nominal_camera() -> PinholeCamera:
    """2236 by 2236 pinhole camera, 90 by 90 degree FOV, pitched 10 degrees down."""
    width, height = square_resolution()
    return PinholeCamera(
        intrinsics=pinhole_intrinsics(width, height, 90.0, 90.0),
        extrinsics=pitched_extrinsics(10),
        frame_rate=20,
        splat_radius=SPLAT_RADIUS,
        feature_output="rgb",
    )
