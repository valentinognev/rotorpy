import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.guidance.camera_geometry import boresight_frd, pitched_extrinsics
from rotorpy.guidance.frames import ned_to_sim
from rotorpy.guidance.homing import (
    HomingCommand,
    HomingGuidance,
    _sightline_ned,
    body_frd_velocity,
    flat_output,
)
from rotorpy.guidance.sightline import ground_relative_position
from rotorpy.guidance.speed import desired_velocity
from rotorpy.guidance.tracking import TrackingState

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
    return HomingGuidance(
        3.0, 36, _course(), INTRINSICS, pitched_extrinsics(10.0), pitched_extrinsics(-70),
    )

def _held_command(velocity_ned, acceleration_ned):
    return HomingCommand(
        velocity_ned,
        acceleration_ned,
        np.zeros(3),
        0.2,
        True,
        True,
        "forward",
        np.array([1.0, 2.0]),
        4.0,
    )


def test_small_velocity_error_is_a_gentle_acceleration():
    """Half a second toward a 0.4 m/s miss is 0.8 m/s², with no extra velocity step."""
    command = _held_command(np.array([10.4, 0.0, 0.0]), np.zeros(3))
    measured = np.array([10.0, 0.0, 0.0])
    flat = flat_output(np.zeros(3), command, measured)
    assert np.allclose(flat["x_dot"], measured)
    assert np.allclose(flat["x_ddot"], [0.8, 0.0, 0.0])
    assert flat["yaw"] == 0.2 and flat["yaw_dot"] == 0.0


def test_steep_velocity_step_stays_inside_an_upright_cone():
    """The opening homing step must pitch, not invert.

    A goal near [19, 9.5, -6] m/s from a slower level flight used to demand
    a downward specific force, and the attitude on that force was the long arc.
    """
    command = _held_command(np.array([19.0, 9.5, 6.0]), np.zeros(3))
    measured = np.array([8.0, 4.0, 0.0])
    flat = flat_output(np.zeros(3), command, measured)
    assert np.allclose(flat["x_dot"], measured)
    specific = flat["x_ddot"] + np.array([0.0, 0.0, 9.81])
    assert specific[2] >= 0.3 * 9.81 - 1e-9
    tilt = np.arctan2(np.linalg.norm(specific[:2]), specific[2])
    assert tilt <= np.deg2rad(45.0) + 1e-9
    miss = ned_to_sim(command.velocity_ned) - measured
    assert np.linalg.norm(np.cross(flat["x_ddot"], miss)) <= 1e-6 * (
        np.linalg.norm(flat["x_ddot"]) * np.linalg.norm(miss)
    )


def test_downward_navigation_cannot_invert_the_specific_force():
    command = _held_command(np.zeros(3), np.array([0.0, 0.0, 100.0]))
    flat = flat_output(np.zeros(3), command, np.zeros(3))
    specific = flat["x_ddot"] + np.array([0.0, 0.0, 9.81])
    assert specific[2] >= 0.3 * 9.81 - 1e-9
    tilt = np.arctan2(np.linalg.norm(specific[:2]), specific[2])
    assert tilt <= np.deg2rad(45.0) + 1e-9


def test_identity_attitude_frd_velocity():
    assert np.allclose(body_frd_velocity(np.array([4.0, 5.0, 6.0]), IDENTITY), [4.0, -5.0, 6.0])

def test_coast_below_lock_and_flat_output():
    guidance = _guidance()
    command = guidance.update(0.0, 30.0, IDENTITY, _course(), np.zeros((32, 32, 3)), np.zeros((32, 32, 3)))
    assert command.locked is False and command.tracking is False
    assert np.allclose(command.velocity_ned, _course())
    assert np.allclose(command.acceleration_ned, 0.0)
    short = _blob(13, 13).copy()
    short[13, 13] = 0.0  # 35 red pixels
    command = guidance.update(0.05, 30.0, IDENTITY, _course(), short, np.zeros((32, 32, 3)))
    assert command.locked is False
    flat = flat_output(
        np.array([1.0, 2.0, 3.0]), command, ned_to_sim(command.velocity_ned)
    )
    assert np.allclose(flat["x"], [1.0, 2.0, 3.0])
    assert np.allclose(flat["x_dot"], ned_to_sim(command.velocity_ned))
    assert flat["yaw_dot"] == 0.0
    assert np.allclose(flat["x_dddot"], 0.0) and np.allclose(flat["x_ddddot"], 0.0)

def test_lock_tracks_boresight_then_holds_through_a_later_sighting():
    guidance = _guidance()
    centered = _blob(13, 13)  # centroid (15.5, 15.5), the principal point
    first = guidance.update(0.0, 30.0, IDENTITY, _course(), centered, np.zeros((32, 32, 3)))
    expected = desired_velocity(ground_relative_position(30.0, boresight_frd(10.0)))
    assert first.locked is True and first.tracking is True
    assert np.allclose(first.acceleration_ned, 0.0)
    assert np.allclose(first.velocity_ned, expected)
    assert np.allclose(first.velocity_body_frd, body_frd_velocity(expected, IDENTITY))
    held = guidance.update(0.01, 30.0, IDENTITY, _course(), None, None)
    assert np.allclose(held.velocity_ned, first.velocity_ned)
    second = guidance.update(0.05, 30.0, IDENTITY, _course(), _blob(17, 13), np.zeros((32, 32, 3)))
    assert second.tracking is True
    assert not np.allclose(second.acceleration_ned, 0.0)
    assert abs(np.dot(second.acceleration_ned, second.velocity_ned)) < 1e-8
    lost = guidance.update(0.10, 30.0, IDENTITY, _course(), np.zeros((32, 32, 3)), np.zeros((32, 32, 3)))
    assert lost.locked is True and lost.tracking is False
    assert np.allclose(lost.velocity_ned, second.velocity_ned)
    assert np.allclose(lost.acceleration_ned, 0.0)
    assert lost.yaw_rad == second.yaw_rad
    again = guidance.update(0.15, 30.0, IDENTITY, _course(), centered, np.zeros((32, 32, 3)))
    assert again.tracking is True and again.source == "forward"

def test_pixels_above_the_horizon_lock_without_tracking():
    guidance = _guidance()
    command = guidance.update(0.0, 30.0, IDENTITY, _course(), _blob(0, 13), np.zeros((32, 32, 3)))
    assert command.locked is True and command.tracking is False
    assert np.allclose(command.velocity_ned, _course())
    assert np.allclose(command.acceleration_ned, 0.0)

def test_coast_yaw_follows_initial_course():
    course = _course()
    guidance = HomingGuidance(
        3.0, 36, course, INTRINSICS, pitched_extrinsics(10.0), pitched_extrinsics(-70),
    )
    command = guidance.update(0.0, 30.0, IDENTITY, course, np.zeros((32, 32, 3)), np.zeros((32, 32, 3)))
    assert command.yaw_rad == np.arctan2(course[1], course[0])


def test_square_center_steers_not_the_centroid():
    guidance = _guidance()
    image = np.zeros((32, 32, 3))
    image[20:26, 10:16, 0] = 1.0
    image[20, 20, 0] = 1.0
    command = guidance.update(0.0, 30.0, IDENTITY, _course(), image, np.zeros((32, 32, 3)))
    assert command.tracking is True and command.source == "forward"
    assert np.allclose(command.tracker_center, [15.0, 22.5])
    assert command.tracker_side == 11
    sightline = _sightline_ned(command.tracker_center, IDENTITY, INTRINSICS, pitched_extrinsics(10.0))
    expected = desired_velocity(ground_relative_position(30.0, sightline))
    assert np.allclose(command.velocity_ned, expected)


def test_upward_latch_survives_a_skyward_sightline_then_steers_nose_down():
    guidance = _guidance()
    black = np.zeros((32, 32, 3))
    blob = _blob(13, 13)
    nose_down = Rotation.from_euler("y", np.deg2rad(80.0)).as_quat()
    sky = guidance.update(0.0, 30.0, IDENTITY, _course(), black, blob)
    assert sky.locked is True and sky.tracking is False and sky.source is None
    assert np.allclose(sky.velocity_ned, _course())
    steered = guidance.update(0.05, 30.0, nose_down, _course(), black, blob)
    assert steered.tracking is True and steered.source == "upward"
    assert np.allclose(steered.acceleration_ned, 0.0)
    assert not np.allclose(steered.velocity_ned, _course())


def test_reset_derivative_history_clears_stale_sightline_and_time():
    # After a vehicle-state clock discontinuity, the previous sightline/timestamp pair
    # came from before the discontinuity and must not be fed into the next line-of-sight
    # rate computation (which would silently fabricate a spurious angular rate from two
    # readings that are not actually adjacent in time).
    guidance = _guidance()
    black = np.zeros((32, 32, 3))
    blob = _blob(13, 13)
    first = guidance.update(0.0, 30.0, IDENTITY, _course(), blob, black)
    shifted = guidance.update(0.05, 30.0, IDENTITY, _course(), _blob(17, 13), black)
    assert shifted.tracking is True and not np.allclose(shifted.acceleration_ned, 0.0)
    guidance.reset_derivative_history()
    after_reset = guidance.update(1000.05, 30.0, IDENTITY, _course(), _blob(17, 13), black)
    # Same image again right after a reset: no stale previous sightline/time survives to
    # produce a (meaningless, huge-dt-driven) acceleration.
    assert np.allclose(after_reset.acceleration_ned, 0.0)


def test_upward_loss_recovers_through_the_forward_camera():
    """Regression for the plan's baseline-failure evidence: a single recorded flight
    (`Results/camera_homing_gazebo.html`) showed the upward camera latch at ~2.14 s,
    lose visibility at ~2.3 s, and *stay* lost for the rest of the flight even though
    the forward camera kept seeing the target the whole time (see
    `tests/test_homing_recorded_handoff_replay.py` for the real recorded frames). The
    old code's one-way `_upward_latched`/`_permanent_loss` pair had no recovery path at
    all once upward latched once. This test replaces
    `test_forward_loss_lets_the_upward_camera_steer_then_loss_sticks`, which asserted
    exactly that permanent-loss behavior as if it were correct.
    """
    guidance = _guidance()
    black = np.zeros((32, 32, 3))
    blob = _blob(13, 13)
    nose_down = Rotation.from_euler("y", np.deg2rad(80.0)).as_quat()
    first = guidance.update(0.0, 30.0, IDENTITY, _course(), blob, black)
    shifted = guidance.update(0.05, 30.0, IDENTITY, _course(), _blob(17, 13), black)
    assert shifted.tracking is True and not np.allclose(shifted.acceleration_ned, 0.0)
    coast = guidance.update(0.10, 30.0, IDENTITY, _course(), black, black)
    assert coast.tracking is False and coast.source is None
    assert np.allclose(coast.velocity_ned, shifted.velocity_ned)
    upward = guidance.update(0.15, 30.0, nose_down, _course(), blob, blob)
    # Both cameras visible: upward (the documented preferred source) latches.
    assert upward.source == "upward" and upward.tracking is True
    assert np.allclose(upward.acceleration_ned, 0.0)  # discontinuous handoff: no stale derivative
    lost = guidance.update(0.20, 30.0, nose_down, _course(), blob, black)
    # Regression: upward disappearing no longer permanently disables guidance -- forward
    # (still visible this very tick) recovers steering immediately, not a stale hold.
    assert lost.tracking is True and lost.source == "forward"
    assert np.allclose(lost.acceleration_ned, 0.0)  # camera handoff: no artificial spike
    assert lost.tracking_state is TrackingState.TRACK
    again = guidance.update(0.25, 30.0, nose_down, _course(), blob, blob)
    # Upward reappearing takes priority back over forward (documented camera preference),
    # itself another handoff -- still just a recoverable camera switch, not a failure.
    assert again.tracking is True and again.source == "upward"


def test_prolonged_loss_aborts_instead_of_holding_a_stale_command_forever():
    """Spec: "Transition to an explicit failure response after prolonged loss" and
    "Prolonged loss does not leave the vehicle in normal stale-command homing." Unlike
    the old permanent-loss hold (a stale velocity command, held forever, indistinguishable
    from a normal in-progress hold), `ABORT` is explicit on the command and one-way.
    """
    guidance = _guidance()
    black = np.zeros((32, 32, 3))
    blob = _blob(13, 13)
    guidance.update(0.0, 30.0, IDENTITY, _course(), blob, black)
    command = None
    # Comfortably past any reasonable proposed max-prediction-gap default.
    for step in range(1, 400):
        command = guidance.update(0.05 * step, 30.0, IDENTITY, _course(), black, black)
        if command.tracking_state is TrackingState.ABORT:
            break
    assert command.tracking_state is TrackingState.ABORT
    assert command.abort_reason is not None
    assert command.tracking is False
    assert np.allclose(command.velocity_ned, 0.0)
    assert np.allclose(command.acceleration_ned, 0.0)
    # One-way: a later sighting must not resurrect a run guidance already gave up on.
    recovered = guidance.update(0.05 * 401, 30.0, IDENTITY, _course(), blob, black)
    assert recovered.tracking_state is TrackingState.ABORT
    assert recovered.tracking is False


def test_complete_is_external_and_stops_steering_without_truth_in_guidance():
    """Spec (controller decision): COMPLETE is a transition the tracker/runner can
    trigger with a reason (the truth-side scorer owns *when*); guidance itself never
    reads truth to decide it."""
    guidance = _guidance()
    blob = _blob(13, 13)
    locked = guidance.update(0.0, 30.0, IDENTITY, _course(), blob, np.zeros((32, 32, 3)))
    assert locked.tracking is True
    guidance.complete("target_reached")
    command = guidance.update(0.05, 30.0, IDENTITY, _course(), blob, np.zeros((32, 32, 3)))
    assert command.tracking_state is TrackingState.COMPLETE
    assert command.tracking is False
    assert np.allclose(command.velocity_ned, 0.0)

