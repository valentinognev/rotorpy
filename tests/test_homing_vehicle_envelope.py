"""Vehicle-envelope/continuity limiting, PN-lead integration, visibility constraint,
and the capture-time/control-time split in `HomingGuidance` (plan Phase 7).

Every behavior here is Gazebo (mounts) path only: the legacy no-mounts path keeps its
exact `desired_velocity`/`pn_acceleration` output (regression-guarded by the existing
`tests/test_homing_command.py`/`tests/test_homing_e2e.py`), since it has no vehicle
envelope to limit against (`HomingGuidance._envelope is None`).

Most tests call the private `_steer`/`update` methods directly with a hand-built
scene (same precedent as `tests/test_homing_command.py`'s own `_sightline_ned`
import): driving the full pixel-detection pipeline to a specific, well-conditioned
`target_motion` status or a specific image-edge pixel position is fragile and
indirect, while `_steer` is exactly the integration point Task 7 changed.
"""

from pathlib import Path

import numpy as np
import pytest

from rotorpy.guidance.homing import HomingCommand
from rotorpy.guidance.homing_config import guidance_from_config, load_homing_config
from rotorpy.guidance.target_estimator import TargetMotionStatus
from rotorpy.guidance.telemetry import setpoint_fields_from_command

CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"
IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def config():
    return load_homing_config(CONFIG_PATH)


def _prime_for_direct_steer(guidance, velocity_ned, t):
    """`update()` always sets these before calling `_steer`; a test calling `_steer`
    directly (bypassing `update()` to control the scene precisely) must set them
    itself, or `_command()`'s own `predict_to` call (using `_last_velocity_ned`) sees
    a `None` velocity and rejects it as non-finite."""
    guidance._last_control_time = t
    guidance._last_velocity_ned = np.array(velocity_ned, dtype=float, copy=True)


def _image_center(guidance):
    return float(guidance.intrinsics["width"]) / 2.0, float(guidance.intrinsics["height"]) / 2.0


def test_mounts_path_total_speed_envelope_is_a_norm_bound_not_a_per_component_one(config):
    """A diagonal direction must not go faster just because each component
    individually fits under a cap (the legacy per-component `desired_velocity`'s
    behavior) -- the new objective bounds the total speed (norm)."""
    guidance = guidance_from_config(config)
    cx, cy = _image_center(guidance)
    relative = np.array([30.0, 20.0, 2.0])
    _prime_for_direct_steer(guidance, np.zeros(3), 0.0)
    command = guidance._steer(
        0.0, IDENTITY, np.zeros(3), "forward", np.array([cx, cy]), 10.0,
        relative / np.linalg.norm(relative), relative,
    )
    assert command.tracking
    assert np.isclose(np.linalg.norm(command.velocity_ned), guidance._envelope.max_speed_m_s, atol=1e-6)


def test_mounts_path_vertical_rate_envelope_bounds_a_steep_aim_point(config):
    guidance = guidance_from_config(config)
    cx, cy = _image_center(guidance)
    relative = np.array([1.0, 1.0, 100.0])  # nearly straight down
    _prime_for_direct_steer(guidance, np.zeros(3), 0.0)
    command = guidance._steer(
        0.0, IDENTITY, np.zeros(3), "forward", np.array([cx, cy]), 10.0,
        relative / np.linalg.norm(relative), relative,
    )
    assert "vertical_rate_envelope" in command.limiting_reasons
    assert abs(command.velocity_ned[2]) <= guidance._envelope.max_vertical_speed_m_s + 1e-9
    assert np.linalg.norm(command.requested_velocity_ned) > np.linalg.norm(command.velocity_ned)


def test_mounts_path_requested_limited_and_reasons_round_trip_through_telemetry(config):
    guidance = guidance_from_config(config)
    cx, cy = _image_center(guidance)
    relative = np.array([1.0, 1.0, 100.0])
    _prime_for_direct_steer(guidance, np.zeros(3), 0.0)
    command = guidance._steer(
        0.0, IDENTITY, np.zeros(3), "forward", np.array([cx, cy]), 10.0,
        relative / np.linalg.norm(relative), relative,
    )
    fields = setpoint_fields_from_command(command)
    assert np.allclose(fields["requested_setpoint"], command.requested_velocity_ned)
    assert np.allclose(fields["limited_setpoint"], command.velocity_ned)
    assert fields["limiting_reasons"] == command.limiting_reasons


def test_legacy_no_mounts_path_never_populates_requested_setpoint_or_reasons():
    command = HomingCommand(
        np.zeros(3), np.zeros(3), np.zeros(3), 0.0, True, True, "forward",
        np.array([4.0, 4.0]), 6.0,
    )
    assert command.requested_velocity_ned is None
    assert setpoint_fields_from_command(command) == {}


def test_mounts_path_falls_back_to_bearing_only_and_records_the_reason_when_unobservable(config):
    """Before the estimator has ever accepted an observation, `target_motion` is
    `UNOBSERVABLE`: steering must fall back to the existing bearing-only aim point
    (no lead) and record why, in `limiting_reasons`."""
    guidance = guidance_from_config(config)
    cx, cy = _image_center(guidance)
    relative = np.array([30.0, 10.0, 2.0])
    _prime_for_direct_steer(guidance, np.zeros(3), 0.0)
    command = guidance._steer(
        0.0, IDENTITY, np.zeros(3), "forward", np.array([cx, cy]), 10.0,
        relative / np.linalg.norm(relative), relative,
    )
    assert command.target_motion is not None
    assert command.target_motion.status is TargetMotionStatus.UNOBSERVABLE
    assert any(reason.startswith("target_motion_unavailable:") for reason in command.limiting_reasons)
    unit_requested = command.requested_velocity_ned / np.linalg.norm(command.requested_velocity_ned)
    unit_relative = relative / np.linalg.norm(relative)
    assert np.allclose(unit_requested, unit_relative, atol=1e-9)


def test_mounts_path_visibility_margin_reduces_speed_near_the_image_edge(config):
    guidance_center = guidance_from_config(config)
    guidance_edge = guidance_from_config(config)
    cx, cy = _image_center(guidance_center)
    width = guidance_center.intrinsics["width"]
    relative = np.array([30.0, 20.0, 2.0])
    sightline = relative / np.linalg.norm(relative)
    _prime_for_direct_steer(guidance_center, np.zeros(3), 0.0)
    _prime_for_direct_steer(guidance_edge, np.zeros(3), 0.0)
    centered = guidance_center._steer(
        0.0, IDENTITY, np.zeros(3), "forward", np.array([cx, cy]), 10.0, sightline, relative
    )
    near_edge = guidance_edge._steer(
        0.0, IDENTITY, np.zeros(3), "forward", np.array([0.02 * width, cy]), 10.0, sightline, relative
    )
    assert "visibility_margin" not in (centered.limiting_reasons or [])
    assert "visibility_margin" in near_edge.limiting_reasons
    assert np.linalg.norm(near_edge.velocity_ned) < np.linalg.norm(centered.velocity_ned)


def test_mounts_path_bounds_the_velocity_step_between_consecutive_ticks(config):
    guidance = guidance_from_config(config)
    cx, cy = _image_center(guidance)
    center = np.array([cx, cy])
    first_relative = np.array([30.0, 20.0, 2.0])
    _prime_for_direct_steer(guidance, np.zeros(3), 0.0)
    first = guidance._steer(
        0.0, IDENTITY, np.zeros(3), "forward", center, 10.0,
        first_relative / np.linalg.norm(first_relative), first_relative,
    )
    # A deliberately huge direction jump one tick (dt=0.01 s) later, at the same
    # (safely centered) pixel position, so only continuity -- not visibility -- is in
    # play.
    second_relative = np.array([-30.0, 25.0, 2.0])
    _prime_for_direct_steer(guidance, np.zeros(3), 0.01)
    second = guidance._steer(
        0.01, IDENTITY, np.zeros(3), "forward", center, 10.0,
        second_relative / np.linalg.norm(second_relative), second_relative,
    )
    step = np.linalg.norm(second.velocity_ned - first.velocity_ned)
    max_step = guidance._envelope.max_accel_m_s2 * 0.01
    assert step <= max_step + 1e-6
    assert "accel_continuity" in second.limiting_reasons


def test_pn_lead_integration_changes_the_transmitted_velocity_with_nav_constant(config):
    """Plan Phase 7 acceptance: changing the navigation constant demonstrably changes
    the transmitted command whenever the target's own estimated motion is nonzero.
    The estimator is primed directly (same pattern `tests/test_target_estimator.py`
    uses) with a genuinely moving target and a stationary vehicle: the full pixel-
    detection pipeline cannot easily produce this (the estimator is specifically
    designed, per Task 6, to cancel drone-induced apparent motion from vehicle
    velocity alone).
    """
    relative_initial = np.array([40.0, 10.0, 15.0])
    true_velocity = np.array([2.0, -1.0, 0.0])
    dt = 0.1

    def _primed(nav_constant):
        guidance = guidance_from_config(config)
        guidance.nav_constant = nav_constant
        cx, cy = _image_center(guidance)
        relative = relative_initial.copy()
        t = 0.0
        for _ in range(60):
            relative = relative + true_velocity * dt
            guidance._estimator.update(
                t=t, vehicle_velocity_ned=np.zeros(3), ray_origin_ned=np.zeros(3),
                ray_direction_ned=relative / np.linalg.norm(relative), height_m=15.0, reference_height_m=0.0,
                fx=guidance.intrinsics["fx"], fy=guidance.intrinsics["fy"], uncertainty_px=1.0,
            )
            t += dt
        _prime_for_direct_steer(guidance, np.zeros(3), t)
        sightline = relative / np.linalg.norm(relative)
        return guidance._steer(t, IDENTITY, np.zeros(3), "forward", np.array([cx, cy]), 10.0, sightline, relative)

    assert _primed(3.0).target_motion.status is TargetMotionStatus.OBSERVABLE
    low = _primed(2.0)
    high = _primed(5.0)
    assert not np.allclose(low.requested_velocity_ned, high.requested_velocity_ned)


def _solid_blob(width, height, column, row, half=6):
    image = np.zeros((height, width, 3), dtype=np.float32)
    r, c = int(round(row)), int(round(column))
    image[r - half:r + half + 1, c - half:c + half + 1, 0] = 1.0
    return image


def test_capture_time_drives_measurement_and_control_time_drives_prediction(config):
    """Carry-over fix: the measurement update (detection/dt/estimator-update) must
    use the image's own capture time; only the target-motion prediction (and
    `_last_control_time`) advances to the later, separate control (processing) time.
    """
    width, height = int(config["image_width_px"]), int(config["image_height_px"])
    cx, cy = width / 2.0, height / 2.0

    no_delay = guidance_from_config(config).update(
        0.0, 30.0, IDENTITY, np.zeros(3), _solid_blob(width, height, cx, cy), None, control_t=0.0
    )
    assert no_delay.target_motion.age_s == pytest.approx(0.0, abs=1e-9)

    delayed = guidance_from_config(config).update(
        0.0, 30.0, IDENTITY, np.zeros(3), _solid_blob(width, height, cx, cy), None, control_t=0.2
    )
    # `predict_to` advanced the estimate to `control_t`, 0.2 s after the one accepted
    # observation at capture time 0.0: age must reflect that gap, not zero.
    assert delayed.target_motion.age_s == pytest.approx(0.2, abs=1e-9)

    # `control_t` defaults to `t` (every pre-Phase-7 single-timestamp call is unaffected).
    defaulted = guidance_from_config(config).update(
        0.0, 30.0, IDENTITY, np.zeros(3), _solid_blob(width, height, cx, cy), None
    )
    assert defaulted.target_motion.age_s == pytest.approx(0.0, abs=1e-9)
