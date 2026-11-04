"""Structured `TargetObservation` wired into the Gazebo (mounts) homing path.

These exercise `HomingGuidance` built with camera mounts (the Gazebo geometry), where
`update()` must consume `pixels.detect_target`/`TargetAssociator` instead of the legacy
`target_square`/`target_pixel_count` functions. The no-mounts (Python-example) path is
untouched and keeps the legacy behavior exercised by `test_homing_command.py`.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rotorpy.guidance.gazebo_bridge import attitude_gazebo_to_homing
from rotorpy.guidance.homing_config import guidance_from_config, load_homing_config
from rotorpy.guidance.pixels import DetectionStatus

CONFIG_PATH = Path(__file__).resolve().parents[1] / "examples" / "camera_homing.jsonc"
NORTH_FACING = Rotation.from_euler("z", 90.0, degrees=True)


@pytest.fixture(scope="module")
def config():
    return load_homing_config(CONFIG_PATH)


def _intrinsics(config):
    width, height = int(config["image_width_px"]), int(config["image_height_px"])
    fx = (width / 2.0) / np.tan(np.deg2rad(config["fov_x_deg"]) / 2.0)
    fy = (height / 2.0) / np.tan(np.deg2rad(config["fov_y_deg"]) / 2.0)
    return fx, fy, width / 2.0, height / 2.0


def _blank(config):
    width, height = int(config["image_width_px"]), int(config["image_height_px"])
    return np.zeros((height, width, 3), dtype=np.float32)


def _solid_blob(config, column, row, half=6):
    image = _blank(config)
    r, c = int(round(row)), int(round(column))
    image[r - half:r + half + 1, c - half:c + half + 1, 0] = 1.0
    return image


def test_tracker_center_and_side_still_populate_on_the_mounts_path(config):
    """The mounts path must still produce a usable tracker square for the preview page."""
    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    guidance = guidance_from_config(config)
    _, _, cx, cy = _intrinsics(config)
    command = guidance.update(0.0, 30.0, attitude, np.zeros(3), _solid_blob(config, cx, cy), None)
    assert command.tracking and command.locked
    assert command.tracker_center is not None
    assert np.allclose(command.tracker_center, [cx, cy], atol=1.0)
    assert command.tracker_side is not None and command.tracker_side > 0


def test_red_distractor_cannot_move_the_locked_reference_across_the_image(config):
    """Once locked, a large red distractor elsewhere must not drag the reference to it.

    The legacy `target_square` bounds every red-like pixel in the frame, so a distractor
    anywhere in the image used to move (or hugely inflate) the tracked square. The
    detector + `TargetAssociator` gate on the predicted location instead.
    """
    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    velocity = np.zeros(3)
    guidance = guidance_from_config(config)
    _, _, cx, cy = _intrinsics(config)

    lock = guidance.update(0.0, 30.0, attitude, velocity, _solid_blob(config, cx + 50.0, cy), None)
    assert lock.tracking and lock.locked

    contaminated = _solid_blob(config, cx + 55.0, cy + 4.0)
    # A distractor blob far across the frame, as strong and large as the real target.
    distractor_half = 40
    d_row, d_col = int(cy + 300.0), int(cx - 400.0)
    contaminated[d_row - distractor_half:d_row + distractor_half + 1,
                 d_col - distractor_half:d_col + distractor_half + 1, 0] = 1.0

    distracted = guidance.update(0.05, 30.0, attitude, velocity, contaminated, None)
    assert distracted.tracking is True
    assert np.linalg.norm(distracted.tracker_center - lock.tracker_center) < 20.0


def test_detections_expose_distinct_statuses_for_no_frame_no_detection_and_ambiguous(config):
    """NO_FRAME, NO_DETECTION and AMBIGUOUS must stay distinguishable on the command,
    not collapse into one undifferentiated 'did not see it' signal."""
    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    velocity = np.zeros(3)

    no_frame_vs_no_detection = guidance_from_config(config).update(
        0.0, 30.0, attitude, velocity, None, _blank(config)
    )
    assert no_frame_vs_no_detection.detections["forward"].status is DetectionStatus.NO_FRAME
    assert no_frame_vs_no_detection.detections["upward"].status is DetectionStatus.NO_DETECTION
    assert no_frame_vs_no_detection.tracking is False and no_frame_vs_no_detection.locked is False

    _, _, cx, cy = _intrinsics(config)
    ambiguous_image = _solid_blob(config, cx - 200.0, cy, half=10)
    other = _solid_blob(config, cx + 200.0, cy, half=10)
    ambiguous_image[other[:, :, 0] > 0] = [1.0, 0.0, 0.0]
    ambiguous = guidance_from_config(config).update(0.0, 30.0, attitude, velocity, ambiguous_image, None)
    assert ambiguous.detections["forward"].status is DetectionStatus.AMBIGUOUS
    assert ambiguous.tracking is False and ambiguous.locked is False


def test_final_approach_grace_coasts_across_a_dropout_a_confident_clipped_lock_would_otherwise_abort(config):
    """Regression for the Gazebo `TARGET_LOST` finding: a large, near-field, clipped
    target right before the nadir blind-spot handoff is a confident lock (its discounted
    `confidence` lands in pixels.py's clipped range, well under 0.5 -- not a weak
    detection), so a brief two-camera dropout during final approach (height at or below
    `homing_config.PROPOSED_FINAL_APPROACH_HEIGHT_M`) must coast past the default
    `max_prediction_gap_s` instead of aborting immediately."""
    from rotorpy.guidance.homing_config import tracking_thresholds
    from rotorpy.guidance.tracking import TrackingState

    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    velocity = np.zeros(3)
    guidance = guidance_from_config(config)
    width, height = int(config["image_width_px"]), int(config["image_height_px"])

    # A large blob that touches the right edge (clipped) and fills most of the frame:
    # the near-field, end-of-approach geometry the grace targets.
    clipped_image = _blank(config)
    clipped_image[height // 4:3 * height // 4, width // 2:, 0] = 1.0
    final_approach_height_m = 15.0
    lock = guidance.update(0.0, final_approach_height_m, attitude, velocity, clipped_image, None)
    assert lock.tracking_state is TrackingState.TRACK
    assert lock.detections["forward"].clipped
    assert 0.0 < lock.detections["forward"].confidence <= 0.5

    default_gap_s = tracking_thresholds()["max_prediction_gap_s"]
    # Past the un-extended bound, inside the final-approach-extended one: must not abort.
    mid_gap_t = default_gap_s + 0.2
    held = guidance.update(mid_gap_t, final_approach_height_m, attitude, velocity, _blank(config), None)
    assert held.tracking_state is not TrackingState.ABORT

    # Past the extended bound too: must abort (the grace is bounded, not infinite). Either
    # reason is acceptable here -- the per-camera sigma also keeps growing during the loss
    # (`TargetAssociator.growth_px_per_s`), so it can cross the uncertainty bound first;
    # what matters is that the grace does not coast forever.
    far_gap_t = default_gap_s * tracking_thresholds().get("final_approach_gap_scale", 2.0) + 0.5
    aborted = guidance.update(far_gap_t, final_approach_height_m, attitude, velocity, _blank(config), None)
    assert aborted.tracking_state is TrackingState.ABORT
    assert aborted.abort_reason in ("max_prediction_gap_exceeded", "max_track_uncertainty_exceeded")


def test_reset_derivative_history_drops_stale_per_camera_tracks(config):
    """After a clock discontinuity, a stale associator track must not gate away a target
    that has genuinely moved past it -- without the reset, `allow_reacquire` still
    recovers a lone candidate eventually, but a concurrent same-frame distractor near the
    stale prediction could otherwise still win; dropping the track removes that risk."""
    attitude = attitude_gazebo_to_homing(NORTH_FACING.as_quat())
    velocity = np.zeros(3)
    guidance = guidance_from_config(config)
    _, _, cx, cy = _intrinsics(config)

    lock = guidance.update(0.0, 30.0, attitude, velocity, _solid_blob(config, cx + 50.0, cy), None)
    assert lock.locked and lock.tracking

    guidance.reset_derivative_history()
    assert guidance._associators["forward"].track is None


def test_no_mounts_legacy_path_has_no_detections_field_populated():
    """The Python-example (no-mounts) path keeps the legacy target_square/_pixel_count
    behavior and does not populate structured detections."""
    from rotorpy.guidance.camera_geometry import pitched_extrinsics
    from rotorpy.guidance.homing import HomingGuidance

    intrinsics = {"fx": 16.0, "fy": 16.0, "width": 32, "height": 32,
                  "cx": 15.5, "cy": 15.5, "dist_coeffs": np.zeros(5)}
    guidance = HomingGuidance(
        3.0, 36, np.array([10.0, 0.0, 0.0]), intrinsics, pitched_extrinsics(10.0), pitched_extrinsics(-70),
    )
    image = np.zeros((32, 32, 3))
    image[13:19, 13:19, 0] = 1.0
    command = guidance.update(0.0, 30.0, np.array([0.0, 0.0, 0.0, 1.0]), np.array([10.0, 0.0, 0.0]),
                              image, np.zeros((32, 32, 3)))
    assert command.detections is None
