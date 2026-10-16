import numpy as np
from rotorpy.guidance.camera_geometry import boresight_frd, pitched_extrinsics
from rotorpy.guidance.frames import ned_to_sim
from rotorpy.guidance.homing import HomingGuidance, body_frd_velocity, flat_output
from rotorpy.guidance.sightline import ground_relative_position
from rotorpy.guidance.speed import desired_velocity

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
INTRINSICS = {"fx": 16.0, "fy": 16.0, "width": 32, "height": 32,
              "cx": 15.5, "cy": 15.5, "dist_coeffs": np.zeros(5)}

def _course():
    direction = np.array([100.0, 50.0, 0.0])
    return 10.0 * direction / np.linalg.norm(direction)

def _blob(row0, col0):
    image = np.zeros((32, 32, 3))
    image[row0:row0 + 6, col0:col0 + 6, 0] = 1.0
    return image

def _guidance():
    return HomingGuidance(3.0, 36, _course(), INTRINSICS, pitched_extrinsics(10.0))

def test_identity_attitude_frd_velocity():
    assert np.allclose(body_frd_velocity(np.array([4.0, 5.0, 6.0]), IDENTITY), [4.0, -5.0, 6.0])

def test_coast_below_lock_and_flat_output():
    guidance = _guidance()
    command = guidance.update(0.0, 30.0, IDENTITY, _course(), np.zeros((32, 32, 3)))
    assert command.locked is False and command.tracking is False
    assert np.allclose(command.velocity_ned, _course())
    assert np.allclose(command.acceleration_ned, 0.0)
    short = _blob(13, 13).copy()
    short[13, 13] = 0.0  # 35 red pixels
    command = guidance.update(0.05, 30.0, IDENTITY, _course(), short)
    assert command.locked is False
    flat = flat_output(np.array([1.0, 2.0, 3.0]), command)
    assert np.allclose(flat["x"], [1.0, 2.0, 3.0])
    assert np.allclose(flat["x_dot"], ned_to_sim(command.velocity_ned))
    assert flat["yaw_dot"] == 0.0
    assert np.allclose(flat["x_dddot"], 0.0) and np.allclose(flat["x_ddddot"], 0.0)

def test_lock_tracks_boresight_then_holds_through_a_later_sighting():
    guidance = _guidance()
    centered = _blob(13, 13)  # centroid (15.5, 15.5), the principal point
    first = guidance.update(0.0, 30.0, IDENTITY, _course(), centered)
    expected = desired_velocity(ground_relative_position(30.0, boresight_frd(10.0)))
    assert first.locked is True and first.tracking is True
    assert np.allclose(first.acceleration_ned, 0.0)
    assert np.allclose(first.velocity_ned, expected)
    assert np.allclose(first.velocity_body_frd, body_frd_velocity(expected, IDENTITY))
    held = guidance.update(0.01, 30.0, IDENTITY, _course(), None)
    assert np.allclose(held.velocity_ned, first.velocity_ned)
    second = guidance.update(0.05, 30.0, IDENTITY, _course(), _blob(17, 13))
    assert second.tracking is True
    assert not np.allclose(second.acceleration_ned, 0.0)
    assert abs(np.dot(second.acceleration_ned, second.velocity_ned)) < 1e-8
    lost = guidance.update(0.10, 30.0, IDENTITY, _course(), np.zeros((32, 32, 3)))
    assert lost.locked is True and lost.tracking is False
    assert np.allclose(lost.velocity_ned, second.velocity_ned)
    assert np.allclose(lost.acceleration_ned, 0.0)
    assert lost.yaw_rad == second.yaw_rad
    again = guidance.update(0.15, 30.0, IDENTITY, _course(), centered)
    assert np.allclose(again.velocity_ned, lost.velocity_ned)
    assert np.allclose(again.acceleration_ned, 0.0)
    assert again.yaw_rad == lost.yaw_rad

def test_pixels_above_the_horizon_lock_without_tracking():
    guidance = _guidance()
    command = guidance.update(0.0, 30.0, IDENTITY, _course(), _blob(0, 13))
    assert command.locked is True and command.tracking is False
    assert np.allclose(command.velocity_ned, _course())
    assert np.allclose(command.acceleration_ned, 0.0)

def test_coast_yaw_follows_initial_course():
    course = _course()
    guidance = HomingGuidance(3.0, 36, course, INTRINSICS, pitched_extrinsics(10.0))
    command = guidance.update(0.0, 30.0, IDENTITY, course, np.zeros((32, 32, 3)))
    assert command.yaw_rad == np.arctan2(course[1], course[0])
