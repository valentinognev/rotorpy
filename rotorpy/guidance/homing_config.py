"""JSONC scene file for the camera-homing example."""

import json
from pathlib import Path

import numpy as np

from rotorpy.guidance.scene import (
    initial_velocity_ned,
    nominal_camera,
    nominal_initial_state,
    nominal_upward_camera,
    nominal_world,
)

CONTROLLERS = ("geometric", "pid")


def strip_jsonc(text: str) -> str:
    """Drop // comments that sit outside strings."""
    out = []
    in_string = False
    escape = False
    i = 0
    while i < len(text):
        ch = text[i]
        if in_string:
            out.append(ch)
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            i += 1
            continue
        if ch == '"':
            in_string = True
            out.append(ch)
            i += 1
            continue
        if ch == "/" and i + 1 < len(text) and text[i + 1] == "/":
            i += 2
            while i < len(text) and text[i] not in "\r\n":
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def load_homing_config(path) -> dict:
    """Parse a camera-homing JSONC file. `controller` is `geometric` or `pid`."""
    config = json.loads(strip_jsonc(Path(path).read_text()))
    controller = config["controller"]
    if controller not in CONTROLLERS:
        raise ValueError(
            "controller must be one of: {}".format(", ".join(CONTROLLERS))
        )
    if config["vehicle"] != "hummingbird":
        raise ValueError("camera homing flies the hummingbird")
    return config


def scene_from_config(config: dict):
    """Initial state, world, forward camera, and upward camera for `config`."""
    start = np.array(config["start_ned"], dtype=float)
    target = np.array(config["target_ground_ned"], dtype=float)
    speed = float(config["initial_speed_m_s"])
    position = np.array(config["camera_position_body_m"], dtype=float)
    camera = dict(
        width=config["image_width_px"],
        height=config["image_height_px"],
        fov_x_deg=config["fov_x_deg"],
        fov_y_deg=config["fov_y_deg"],
        position=position,
        frame_rate=config["camera_rate_hz"],
        splat_radius=config["splat_radius_px"],
        feature_output=config["feature_output"],
    )
    state = nominal_initial_state(
        start_ned=start,
        target_ground_ned=target,
        initial_speed=speed,
        body_rate=config["initial_body_rate_rad_s"],
        wind=config["initial_wind_m_s"],
    )
    world = nominal_world(
        bounds=config["world_bounds_m"],
        extents=config["box_extents_m"],
        color=config["box_color_rgb"],
        feature_spacing=config["feature_spacing_m"],
        feature_mode=config["feature_mode"],
        descriptor_noise=config["descriptor_noise"],
    )
    forward = nominal_camera(pitch_deg=config["forward_pitch_deg"], **camera)
    upward = nominal_upward_camera(pitch_deg=config["upward_pitch_deg"], **camera)
    return state, world, forward, upward


def initial_velocity_from_config(config: dict) -> np.ndarray:
    """Horizontal NED course stored by the homing law before lock."""
    return initial_velocity_ned(
        np.array(config["start_ned"], dtype=float),
        np.array(config["target_ground_ned"], dtype=float),
        float(config["initial_speed_m_s"]),
    )
