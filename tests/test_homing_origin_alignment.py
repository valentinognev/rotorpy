"""Estimator-origin and simulator-origin alignment."""

import numpy as np
import pytest

from rotorpy.guidance.homing_config import alignment_from_config, load_homing_config
from rotorpy.guidance.origin_alignment import EstimatorOriginAlignment

from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"


def _spawn_aligned():
    return EstimatorOriginAlignment(origin_world_ned=[0.0, 0.0, -30.0], yaw_offset_rad=0.0, ground_down_world_m=0.0)


def test_estimator_origin_at_the_spawn_point_shifts_world_coordinates():
    alignment = _spawn_aligned()
    assert np.allclose(alignment.world_to_estimator([100.0, 50.0, 0.0]), [100.0, 50.0, 30.0])
    assert np.allclose(alignment.estimator_to_world([0.0, 0.0, 0.0]), [0.0, 0.0, -30.0])


def test_world_and_estimator_coordinates_differ_when_the_origin_is_not_shared():
    alignment = EstimatorOriginAlignment([10.0, -5.0, -2.0], 0.0, 0.0)
    world = np.array([4.0, 6.0, -12.0])
    assert not np.allclose(alignment.world_to_estimator(world), world)
    assert np.allclose(alignment.estimator_to_world(alignment.world_to_estimator(world)), world)


def test_vectors_only_rotate_never_translate():
    alignment = EstimatorOriginAlignment([10.0, -5.0, -2.0], np.deg2rad(90.0), 0.0)
    assert np.allclose(alignment.world_vector_to_estimator([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-12)
    assert np.allclose(alignment.estimator_vector_to_world([0.0, -1.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(alignment.world_vector_to_estimator([0.0, 0.0, 3.0]), [0.0, 0.0, 3.0])


def test_yaw_offset_rotates_positions_about_the_down_axis_and_round_trips():
    alignment = EstimatorOriginAlignment([1.0, 2.0, -30.0], np.deg2rad(30.0), 0.0)
    estimator = np.array([40.0, -7.0, 12.0])
    world = alignment.estimator_to_world(estimator)
    assert np.isclose(np.linalg.norm(world[:2] - [1.0, 2.0]), np.linalg.norm(estimator[:2]))
    assert np.isclose(world[2], -30.0 + 12.0)
    assert np.allclose(alignment.world_to_estimator(world), estimator)
    # Estimator north points 30 degrees clockwise of world north.
    assert np.allclose(alignment.estimator_to_world([1.0, 0.0, 0.0]) - [1.0, 2.0, -30.0],
                       [np.cos(np.deg2rad(30.0)), np.sin(np.deg2rad(30.0)), 0.0])


def test_heading_conversion_uses_the_same_offset():
    alignment = EstimatorOriginAlignment([0.0, 0.0, 0.0], np.deg2rad(30.0), 0.0)
    assert np.isclose(alignment.heading_to_world(np.deg2rad(10.0)), np.deg2rad(40.0))
    assert np.isclose(alignment.heading_to_estimator(np.deg2rad(40.0)), np.deg2rad(10.0))


def test_height_above_ground_uses_the_ground_plane_not_the_estimator_origin():
    alignment = _spawn_aligned()
    assert np.isclose(alignment.height_above_ground([5.0, 5.0, 0.0]), 30.0)
    assert np.isclose(alignment.height_above_ground([5.0, 5.0, 30.0]), 0.0)
    raised_ground = EstimatorOriginAlignment([0.0, 0.0, -30.0], 0.0, ground_down_world_m=-2.0)
    assert np.isclose(raised_ground.height_above_ground([0.0, 0.0, 0.0]), 28.0)
    assert np.isclose(raised_ground.ground_down_in_estimator_m, 28.0)


def test_non_finite_inputs_are_rejected():
    with pytest.raises(ValueError):
        EstimatorOriginAlignment([np.nan, 0.0, 0.0], 0.0, 0.0)
    with pytest.raises(ValueError):
        EstimatorOriginAlignment([0.0, 0.0], 0.0, 0.0)
    with pytest.raises(ValueError):
        _spawn_aligned().world_to_estimator([np.inf, 0.0, 0.0])


def test_alignment_comes_from_the_config_and_matches_the_spawn():
    config = load_homing_config(CONFIG_PATH)
    alignment = alignment_from_config(config)
    assert np.allclose(alignment.origin_world_ned, config["start_ned"])
    assert alignment.yaw_offset_rad == 0.0
    assert np.isclose(alignment.height_above_ground([0.0, 0.0, 0.0]), -config["start_ned"][2])
