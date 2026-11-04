"""Unit tests for the offline reliability harness's moving-target world builder and
target-motion profiles (plan Phase 9)."""

import numpy as np
import pytest

from rotorpy.guidance.frames import ned_to_sim
from rotorpy.guidance.homing_config import PROPOSED_MAX_TARGET_ACCEL_M_S2, PROPOSED_MAX_TARGET_SPEED_M_S
from rotorpy.guidance.validation.harness_config import harness_config
from rotorpy.guidance.validation.target_world import MovingTargetWorldBuilder, TargetMotionProfile


@pytest.fixture(scope="module")
def config():
    return harness_config()


def test_stationary_profile_never_moves():
    profile = TargetMotionProfile(kind="stationary", start_ground_ned=np.array([10.0, 5.0, 0.0]))
    for t in (0.0, 1.0, 20.0):
        assert np.allclose(profile.position_ned(t), [10.0, 5.0, 0.0])


def test_crossing_profile_moves_at_the_configured_speed():
    profile = TargetMotionProfile(
        kind="crossing", start_ground_ned=np.zeros(3), speed_m_s=2.0, heading_rad=0.0,
    )
    p0 = profile.position_ned(0.0)
    p1 = profile.position_ned(1.0)
    assert np.isclose(np.linalg.norm(p1 - p0), 2.0)
    assert np.allclose(p1, [2.0, 0.0, 0.0])


def test_turning_profile_moves_at_constant_speed_along_a_curved_path():
    profile = TargetMotionProfile(
        kind="turning", start_ground_ned=np.zeros(3), speed_m_s=1.0, heading_rad=0.0, turn_rate_rad_s=0.5,
    )
    dt = 1e-3
    times = np.linspace(0.0, 2.0, 50)
    positions = [profile.position_ned(t) for t in times]
    speeds = [np.linalg.norm(profile.position_ned(t + dt) - profile.position_ned(t)) / dt for t in times]
    assert all(np.isclose(speed, 1.0, atol=1e-3) for speed in speeds)
    # A circular arc: distinct positions over time, and never a straight line.
    assert len(set(tuple(np.round(p, 6)) for p in positions)) == len(positions)
    assert not np.allclose(positions[0][:2], positions[-1][:2])


def test_accelerating_profile_speed_saturates_at_the_target_envelope():
    profile = TargetMotionProfile(
        kind="accelerating", start_ground_ned=np.zeros(3), speed_m_s=0.0, heading_rad=0.0,
        accel_m_s2=PROPOSED_MAX_TARGET_ACCEL_M_S2,
    )
    dt = 1e-3
    far_future = 1000.0
    p0 = profile.position_ned(far_future)
    p1 = profile.position_ned(far_future + dt)
    speed = np.linalg.norm(p1 - p0) / dt
    assert speed <= PROPOSED_MAX_TARGET_SPEED_M_S + 1e-6


def test_profile_rejects_speed_above_the_proposed_envelope():
    with pytest.raises(ValueError):
        TargetMotionProfile(
            kind="crossing", start_ground_ned=np.zeros(3),
            speed_m_s=PROPOSED_MAX_TARGET_SPEED_M_S + 1.0, heading_rad=0.0,
        )


def test_profile_rejects_lateral_acceleration_above_the_proposed_envelope():
    with pytest.raises(ValueError):
        TargetMotionProfile(
            kind="turning", start_ground_ned=np.zeros(3), speed_m_s=5.0, heading_rad=0.0,
            turn_rate_rad_s=10.0,
        )


def test_moving_target_world_translates_features_by_the_ground_delta(config):
    builder = MovingTargetWorldBuilder(config)
    reference = np.asarray(config["target_ground_ned"], dtype=float)
    shift_ned = np.array([3.0, -2.0, 0.0])
    world_reference = builder.world_at(reference)
    world_shifted = builder.world_at(reference + shift_ned)
    features_reference = world_reference.get_surface_features()
    features_shifted = world_shifted.get_surface_features()
    assert features_reference.shape == features_shifted.shape
    assert features_reference.shape[0] > 0
    delta_sim = ned_to_sim(shift_ned)
    assert np.allclose(features_shifted - features_reference, delta_sim, atol=1e-9)


def test_moving_target_world_block_extents_track_the_target(config):
    builder = MovingTargetWorldBuilder(config)
    reference = np.asarray(config["target_ground_ned"], dtype=float)
    shift_ned = np.array([0.0, 4.0, 0.0])
    extents_reference = builder.world_at(reference).get_block_bounding_boxes()[0]
    extents_shifted = builder.world_at(reference + shift_ned).get_block_bounding_boxes()[0]
    delta_sim = ned_to_sim(shift_ned)
    expected = [
        extents_reference[0] + delta_sim[0], extents_reference[1] + delta_sim[0],
        extents_reference[2] + delta_sim[1], extents_reference[3] + delta_sim[1],
        extents_reference[4] + delta_sim[2], extents_reference[5] + delta_sim[2],
    ]
    assert np.allclose(extents_shifted, expected, atol=1e-9)
