"""Truth-side target motion profiles and a cheap moving-target `World` (plan Phase 9).

`TargetMotionProfile` is ground truth only: it is never read by guidance (guidance
only ever sees camera pixels, exactly as in production). `MovingTargetWorldBuilder`
turns a profile's ground position into a `World` the project's own `PinholeCamera`
renders, reusing the exact same surface-feature generator `scene.nominal_world`
uses for the Python example's static target -- but only once, at construction, to
build a feature *template*. Every later `world_at(...)` call rigidly translates that
template (an embedded-features `World`, see `rotorpy.world_features`'s "Embedded
features (bypass the generator entirely, no RNG involved)" case) instead of
regenerating it, which is the difference between a per-trial setup cost and a
per-tick one over a closed-loop run of thousands of ticks.
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from rotorpy.guidance.frames import ned_to_sim
from rotorpy.guidance.homing_config import (
    PROPOSED_MAX_TARGET_ACCEL_M_S2,
    PROPOSED_MAX_TARGET_SPEED_M_S,
    gazebo_target_extents_sim,
)
from rotorpy.guidance.scene import nominal_world
from rotorpy.world import World

PROFILE_KINDS = ("stationary", "crossing", "approaching", "receding", "turning", "accelerating")


@dataclass(frozen=True)
class TargetMotionProfile:
    """Ground-truth target motion, bounded by the guidance's own proposed envelope
    (`homing_config.PROPOSED_MAX_TARGET_SPEED_M_S`/`_ACCEL_M_S2`): a scenario whose
    target moves faster or harder than the envelope the estimator (`target_estimator.
    TargetMotionEstimator`) was configured to trust is not a fair reliability trial of
    that estimator, so out-of-envelope parameters are rejected at construction.

    `kind` is one of `PROFILE_KINDS`. `heading_rad` is the ground course for
    `crossing`/`approaching`/`receding`/`turning` (a straight line for the first
    three; the initial heading for `turning`, which then curves at `turn_rate_rad_s`).
    `accelerating` starts at `speed_m_s` along `heading_rad` and speeds up at
    `accel_m_s2` until it reaches the envelope's own max speed, then holds it.
    """

    kind: str
    start_ground_ned: np.ndarray
    speed_m_s: float = 0.0
    heading_rad: float = 0.0
    turn_rate_rad_s: float = 0.0
    accel_m_s2: float = 0.0

    def __post_init__(self):
        if self.kind not in PROFILE_KINDS:
            raise ValueError("kind must be one of {}".format(PROFILE_KINDS))
        if abs(self.speed_m_s) > PROPOSED_MAX_TARGET_SPEED_M_S + 1e-9:
            raise ValueError("speed_m_s exceeds the proposed target-motion envelope")
        if abs(self.accel_m_s2) > PROPOSED_MAX_TARGET_ACCEL_M_S2 + 1e-9:
            raise ValueError("accel_m_s2 exceeds the proposed target-motion envelope")
        lateral_accel = abs(self.speed_m_s * self.turn_rate_rad_s)
        if lateral_accel > PROPOSED_MAX_TARGET_ACCEL_M_S2 + 1e-9:
            raise ValueError("speed_m_s * turn_rate_rad_s exceeds the proposed target-motion envelope")

    def _accelerating_displacement(self, t_s: float) -> float:
        if self.accel_m_s2 <= 0.0:
            return self.speed_m_s * t_s
        t_cap = max(0.0, (PROPOSED_MAX_TARGET_SPEED_M_S - self.speed_m_s) / self.accel_m_s2)
        if t_s <= t_cap:
            return self.speed_m_s * t_s + 0.5 * self.accel_m_s2 * t_s * t_s
        capped = self.speed_m_s * t_cap + 0.5 * self.accel_m_s2 * t_cap * t_cap
        return capped + PROPOSED_MAX_TARGET_SPEED_M_S * (t_s - t_cap)

    def position_ned(self, t_s: float) -> np.ndarray:
        """Ground-contact NED position at `t_s` (`t_s >= 0`)."""
        t_s = float(t_s)
        start = np.asarray(self.start_ground_ned, dtype=float)
        if self.kind == "stationary":
            return start.copy()
        if self.kind == "turning" and abs(self.turn_rate_rad_s) > 1e-9:
            omega = self.turn_rate_rad_s
            radius_speed = self.speed_m_s / omega
            theta0 = self.heading_rad
            theta = theta0 + omega * t_s
            dx = radius_speed * (np.sin(theta) - np.sin(theta0))
            dy = -radius_speed * (np.cos(theta) - np.cos(theta0))
            return start + np.array([dx, dy, 0.0])
        if self.kind == "accelerating":
            displacement = self._accelerating_displacement(t_s)
        else:
            displacement = self.speed_m_s * t_s
        direction = np.array([np.cos(self.heading_rad), np.sin(self.heading_rad), 0.0])
        return start + direction * displacement


def _shift_extents(extents_sim, delta_sim) -> list:
    xmin, xmax, ymin, ymax, zmin, zmax = (float(v) for v in extents_sim)
    dx, dy, dz = (float(v) for v in delta_sim)
    return [xmin + dx, xmax + dx, ymin + dy, ymax + dy, zmin + dz, zmax + dz]


class MovingTargetWorldBuilder(object):
    """Cheap, per-tick `World` for a target that may be moving.

    Builds the feature template once with the project's own regular-grid surface-
    feature generator (`scene.nominal_world`, the same one the static Python-example
    target uses) at the config's reference target position, then every `world_at`
    call rigidly translates that template by the target's ground-position delta --
    no RNG, no regeneration, on the hot path.
    """

    def __init__(self, config: dict) -> None:
        self._reference_ground_ned = np.asarray(config["target_ground_ned"], dtype=float)
        self._reference_extents_sim = list(gazebo_target_extents_sim(config))
        self._bounds = [float(value) for value in config["world_bounds_m"]]
        self._color = [float(value) for value in config["box_color_rgb"]]
        reference_world = nominal_world(
            bounds=self._bounds,
            extents=self._reference_extents_sim,
            color=self._color,
            feature_spacing=config["feature_spacing_m"],
            feature_mode=config["feature_mode"],
            descriptor_noise=config["descriptor_noise"],
        )
        features = reference_world.get_surface_features()
        self._base_features = np.zeros((0, 3)) if features is None else np.array(features, dtype=float, copy=True)
        colors = reference_world.get_feature_colors()
        self._base_colors = None if colors is None else np.array(colors, dtype=float, copy=True)

    def world_at(self, ground_contact_ned) -> World:
        """A `World` with the target shifted to `ground_contact_ned`."""
        delta_ned = np.asarray(ground_contact_ned, dtype=float) - self._reference_ground_ned
        delta_sim = ned_to_sim(delta_ned)
        extents = _shift_extents(self._reference_extents_sim, delta_sim)
        features = self._base_features + delta_sim if self._base_features.size else self._base_features
        features_config = {"points": features}
        if self._base_colors is not None:
            features_config["colors"] = self._base_colors
        world_data = {
            "bounds": {"extents": self._bounds},
            "blocks": [{"extents": extents, "color": self._color}],
            "features": features_config,
        }
        return World(world_data)
