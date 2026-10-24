"""The camera-homing example reads examples/camera_homing.jsonc."""

from pathlib import Path

import numpy as np

from rotorpy.controllers.px4_cascade import Px4Cascade
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.guidance.homing_config import load_homing_config, scene_from_config
from rotorpy.guidance.runner import make_controller
from rotorpy.guidance.scene import (
    INITIAL_SPEED,
    START_NED,
    TARGET_GROUND_NED,
    nominal_initial_state,
)
from rotorpy.vehicles.hummingbird_params import quad_params

_CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"


def _parameter_lines(text):
    """Each JSON key whose previous non-empty line is a // comment."""
    described = []
    previous = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith('"') and '":' in stripped:
            key = stripped.split('"', 2)[1]
            described.append((key, previous.startswith("//")))
        if stripped:
            previous = stripped
    return described


def test_jsonc_comments_every_parameter_and_matches_the_nominal_scene():
    text = _CONFIG_PATH.read_text()
    described = _parameter_lines(text)
    assert described
    assert all(has_comment for _, has_comment in described)
    config = load_homing_config(_CONFIG_PATH)
    assert config["controller"] == "pid"
    assert "geometric" in text and "pid" in text
    assert np.allclose(config["start_ned"], START_NED)
    assert np.allclose(config["target_ground_ned"], TARGET_GROUND_NED)
    assert config["initial_speed_m_s"] == INITIAL_SPEED
    assert np.allclose(config["box_center_ned"], [100.0, 50.0, -0.5])
    state, _, _, _ = scene_from_config(config)
    nominal = nominal_initial_state()
    assert np.allclose(state["x"], nominal["x"])
    assert np.allclose(state["v"], nominal["v"])
    assert np.allclose(state["q"], nominal["q"])
    assert np.allclose(state["w"], nominal["w"])
    assert np.allclose(state["wind"], nominal["wind"])
    assert np.allclose(state["rotor_speeds"], nominal["rotor_speeds"])


def test_controller_field_selects_geometric_or_pid():
    assert isinstance(make_controller("pid", quad_params), Px4Cascade)
    assert isinstance(make_controller("geometric", quad_params), SE3Control)
