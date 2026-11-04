import base64
import io
from pathlib import Path

import numpy as np
from rotorpy.guidance.pixels import target_centroid, target_pixel_count, target_square

_GAZEBO_HTML_PATH = Path(__file__).resolve().parents[1] / "Results" / "camera_homing_gazebo.html"


def _opening_forward_frame():
    """The first forward-camera frame embedded in `Results/camera_homing_gazebo.html`.

    This is the plan's documented baseline-failure evidence: "The opening recorded
    square spans almost the entire preview although the target is small." Decoded here
    from the actual recorded JPEG rather than synthesized, per the task-4 brief.
    """
    from PIL import Image

    content = _GAZEBO_HTML_PATH.read_text()
    marker = '"forward_image": "data:image/jpeg;base64,'
    start = content.index(marker) + len(marker)
    end = content.index('"', start)
    raw = base64.b64decode(content[start:end])
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    return np.asarray(image, dtype=float) / 255.0


def test_opening_failure_frame_selects_the_cylinder_not_an_image_wide_region():
    from rotorpy.guidance.pixels import DetectionStatus, detect_target

    image = _opening_forward_frame()
    height, width = image.shape[:2]

    old = target_square(image)
    assert old is not None
    # The baseline-failure evidence: the legacy bounding square spans nearly the whole
    # frame even though the recorded target cylinder is small.
    assert old[2] >= 0.9 * width

    obs = detect_target(image, timestamp=0.0, camera="forward")
    assert obs.status is DetectionStatus.DETECTED
    assert obs.area < 0.01 * height * width
    extent = max(obs.bounds[2] - obs.bounds[0], obs.bounds[3] - obs.bounds[1]) + 1
    assert extent < 0.1 * width


def test_empty_image_has_no_target():
    image = np.zeros((4, 4, 3))
    assert target_pixel_count(image) == 0
    assert target_centroid(image) is None

def test_red_threshold_and_centroid():
    image = np.zeros((4, 5, 3))
    image[1, 2] = [0.51, 0.5, 0.5]
    image[1, 3] = [0.5, 0.0, 0.0]       # red == 0.5 does not count
    image[2, 2] = [0.9, 0.9, 0.0]       # red is not strictly greater than green
    assert target_pixel_count(image) == 1
    assert np.allclose(target_centroid(image), [2.0, 1.0])

def test_centroid_is_mean_of_every_target_pixel():
    image = np.zeros((4, 5, 3))
    image[0, 0] = [0.8, 0.0, 0.0]
    image[2, 4] = [0.9, 0.1, 0.1]
    assert target_pixel_count(image) == 2
    assert np.allclose(target_centroid(image), [2.0, 1.0])

def test_red_equal_blue_is_not_a_target():
    image = np.zeros((3, 3, 3))
    image[1, 1] = [0.8, 0.1, 0.8]  # red == blue, red > 0.5, red > green
    assert target_pixel_count(image) == 0
    assert target_centroid(image) is None

def test_target_square_is_the_bounding_square():
    image = np.zeros((32, 32, 3))
    assert target_square(image) is None
    image[20:26, 10:16, 0] = 1.0
    image[20, 20, 0] = 1.0
    center_u, center_v, side = target_square(image)
    assert (center_u, center_v, side) == (15.0, 22.5, 11)
    assert not np.allclose([center_u, center_v], target_centroid(image))


from dataclasses import FrozenInstanceError

import pytest

from rotorpy.guidance.pixels import (
    DetectionStatus,
    TargetAssociator,
    TrackPrediction,
    detect_target,
)

RED = [0.85, 0.1, 0.1]


def _background(height=96, width=128):
    image = np.empty((height, width, 3))
    image[:] = [0.35, 0.5, 0.6]
    return image


def _blob(image, row, col, height, width, color=RED):
    image[row:row + height, col:col + width] = color


def _contaminate(image, seed=0):
    rng = np.random.default_rng(seed)
    height, width = image.shape[:2]
    weak = rng.random((height, width)) < 0.06
    image[weak] = [0.6, 0.47, 0.47]
    reddish = rng.random((height, width)) < 0.01
    image[reddish] = [0.8, 0.2, 0.2]


def test_contaminated_frame_selects_the_blob_not_an_image_wide_region():
    image = _background()
    _blob(image, 50, 80, 24, 12)
    _contaminate(image)
    old = target_square(image)
    assert old[2] > 90  # the legacy bounding square spans the picture
    obs = detect_target(image, timestamp=1.5, camera="forward")
    assert obs.status is DetectionStatus.DETECTED
    # a scattered strong pixel touching the blob may extend it by one pixel
    assert np.allclose(obs.bounds, (80, 50, 91, 73), atol=1)
    assert np.allclose(obs.reference, [85.5, 61.5], atol=0.5)
    assert 0.8 * 24 * 12 <= obs.area <= 1.2 * 24 * 12
    assert not obs.clipped
    assert obs.timestamp == 1.5 and obs.camera == "forward"
    assert 0.0 < obs.confidence <= 1.0
    assert 0.0 < obs.uncertainty_px < 5.0


def test_weak_red_dominance_is_not_a_target():
    image = _background()
    _blob(image, 40, 40, 10, 10, [0.6, 0.5, 0.5])
    assert detect_target(image).status is DetectionStatus.NO_DETECTION


def test_small_distant_target_is_preserved():
    image = _background()
    _blob(image, 30, 70, 3, 3)
    obs = detect_target(image)
    assert obs.status is DetectionStatus.DETECTED
    assert obs.bounds == (70, 30, 72, 32)
    assert np.allclose(obs.reference, [71.0, 31.0])
    assert obs.area == 9


def test_small_target_survives_contamination():
    image = _background()
    _blob(image, 30, 70, 3, 3)
    _contaminate(image, seed=3)
    obs = detect_target(image)
    assert obs.status is DetectionStatus.DETECTED
    assert np.allclose(obs.reference, [71.0, 31.0], atol=0.5)


def test_single_pixels_and_thin_lines_are_rejected():
    image = _background()
    image[10, 10] = RED
    image[50, 20:60] = RED
    assert detect_target(image).status is DetectionStatus.NO_DETECTION


def test_no_frame_no_detection_and_ambiguous_are_distinct():
    assert detect_target(None, timestamp=2.0, camera="up").status is DetectionStatus.NO_FRAME
    empty = detect_target(_background(), timestamp=2.0, camera="up")
    assert empty.status is DetectionStatus.NO_DETECTION
    assert empty.reference is None and empty.confidence == 0.0
    image = _background()
    _blob(image, 20, 20, 12, 12)
    _blob(image, 60, 90, 12, 12)
    amb = detect_target(image)
    assert amb.status is DetectionStatus.AMBIGUOUS
    assert amb.reference is None


def test_dominant_component_wins_without_a_track():
    image = _background()
    _blob(image, 20, 20, 30, 15)
    _blob(image, 70, 100, 4, 4)
    obs = detect_target(image)
    assert obs.status is DetectionStatus.DETECTED
    assert np.allclose(obs.reference, [27.0, 34.5])


def test_clipped_only_candidate_is_flagged_with_inflated_uncertainty():
    inside = _background()
    _blob(inside, 40, 50, 20, 10)
    clipped = _background()
    _blob(clipped, 40, 118, 20, 10)  # runs off the right edge
    base = detect_target(inside)
    obs = detect_target(clipped)
    assert obs.status is DetectionStatus.DETECTED
    assert obs.clipped and not base.clipped
    assert obs.bounds[2] == 127
    assert obs.uncertainty_px > base.uncertainty_px
    assert obs.confidence < base.confidence


def test_large_near_field_clipped_target_uncertainty_stays_bounded():
    from rotorpy.guidance.homing import _MAX_TRACK_UNCERTAINTY_PX

    # A big near-field target that fills (and clips) most of a real-size frame on final
    # approach is a confident lock, not a highly uncertain one. Its raw bounding-box side
    # (300+ px) must not inflate uncertainty_px past the track-abort threshold, or a single
    # missed frame right after a good detection aborts the homing run (the Gazebo
    # `TARGET_LOST` failure). The saturating extent cap keeps it safely bounded.
    image = _background(height=360, width=360)
    _blob(image, 100, 0, 150, 360)  # 360 px wide, runs to both side edges -> clipped
    obs = detect_target(image)
    assert obs.status is DetectionStatus.DETECTED
    assert obs.clipped
    assert obs.uncertainty_px < _MAX_TRACK_UNCERTAINTY_PX


def test_unclipped_candidate_is_preferred_over_a_slightly_larger_clipped_one():
    image = _background()
    _blob(image, 40, 118, 20, 10)
    _blob(image, 10, 30, 18, 10)
    obs = detect_target(image)
    assert obs.status is DetectionStatus.DETECTED
    assert not obs.clipped
    assert np.allclose(obs.reference, [34.5, 18.5])


def test_prediction_gate_selects_the_consistent_component():
    image = _background()
    _blob(image, 20, 20, 12, 12)
    _blob(image, 60, 90, 12, 12)
    near_first = detect_target(image, prediction=TrackPrediction(24.0, 27.0, 5.0))
    near_second = detect_target(image, prediction=TrackPrediction(94.0, 67.0, 5.0))
    assert near_first.status is DetectionStatus.DETECTED
    assert np.allclose(near_first.reference, [25.5, 25.5])
    assert near_second.status is DetectionStatus.DETECTED
    assert np.allclose(near_second.reference, [95.5, 65.5])


def test_larger_distractor_outside_the_gate_is_ignored():
    image = _background()
    _blob(image, 20, 20, 8, 8)
    _blob(image, 50, 80, 30, 25)
    obs = detect_target(image, prediction=TrackPrediction(24.0, 24.0, 4.0))
    assert obs.status is DetectionStatus.DETECTED
    assert np.allclose(obs.reference, [23.5, 23.5])


def test_distractor_alone_outside_the_gate_does_not_move_the_reference():
    image = _background()
    _blob(image, 50, 80, 30, 25)
    obs = detect_target(image, prediction=TrackPrediction(24.0, 24.0, 4.0))
    assert obs.status is DetectionStatus.NO_DETECTION
    assert obs.reference is None


def test_two_components_equally_consistent_with_the_gate_are_ambiguous():
    image = _background()
    _blob(image, 40, 30, 10, 10)
    _blob(image, 40, 60, 10, 10)
    obs = detect_target(image, prediction=TrackPrediction(50.0, 45.0, 30.0))
    assert obs.status is DetectionStatus.AMBIGUOUS


def test_associator_holds_the_track_against_a_distractor():
    associator = TargetAssociator()
    first = _background()
    _blob(first, 20, 20, 8, 8)
    assert associator.update(first, 0.0, "forward").status is DetectionStatus.DETECTED

    second = _background()
    _blob(second, 22, 24, 8, 8)
    _blob(second, 50, 80, 12, 12)
    obs = associator.update(second, 0.05, "forward")
    assert obs.status is DetectionStatus.DETECTED
    assert np.allclose(obs.reference, [27.5, 25.5])

    third = _background()
    _blob(third, 50, 80, 12, 12)
    lost = associator.update(third, 0.10, "forward")
    assert lost.status is DetectionStatus.NO_DETECTION
    assert associator.track is not None
    assert np.allclose([associator.track.u, associator.track.v], [27.5, 25.5])


def test_associator_gate_widens_while_the_target_is_missing():
    associator = TargetAssociator()
    first = _background()
    _blob(first, 20, 20, 8, 8)
    associator.update(first, 0.0, "forward")
    sigma = associator.track.sigma_px
    associator.update(_background(), 1.0, "forward")
    assert associator.track.sigma_px > sigma
    again = _background()
    _blob(again, 40, 50, 8, 8)
    assert associator.update(again, 1.0, "forward").status is DetectionStatus.DETECTED


def test_associator_ignores_the_track_from_another_camera_and_can_reset():
    associator = TargetAssociator()
    first = _background()
    _blob(first, 20, 20, 8, 8)
    associator.update(first, 0.0, "forward")
    other = _background()
    _blob(other, 60, 90, 8, 8)
    assert associator.update(other, 0.05, "up").status is DetectionStatus.DETECTED
    assert associator.update(None, 0.1, "up").status is DetectionStatus.NO_FRAME
    associator.reset()
    assert associator.track is None


def test_allow_reacquire_recovers_a_single_candidate_outside_the_gate():
    """`allow_reacquire=True` falls back to an ungated search when the gate rejects
    everything, so a legitimate large jump is not permanently lost -- but the default
    (`allow_reacquire=False`, every other test above) keeps rejecting it."""
    first = _background()
    _blob(first, 20, 20, 8, 8)
    moved = _background()
    _blob(moved, 70, 90, 8, 8)

    default_associator = TargetAssociator()
    default_associator.update(first, 0.0, "forward")
    default = default_associator.update(moved, 0.05, "forward")
    assert default.status is DetectionStatus.NO_DETECTION

    reacquiring_associator = TargetAssociator()
    reacquiring_associator.update(first, 0.0, "forward")
    recovered = reacquiring_associator.update(moved, 0.05, "forward", allow_reacquire=True)
    assert recovered.status is DetectionStatus.DETECTED
    assert np.allclose(recovered.reference, [93.5, 73.5])


def test_allow_reacquire_still_prefers_the_gated_candidate_over_a_distractor():
    """A real distractor present in the *same* frame as the gated candidate must still
    lose to it, even with `allow_reacquire=True`: the gate is checked first."""
    associator = TargetAssociator()
    first = _background()
    _blob(first, 20, 20, 8, 8)
    associator.update(first, 0.0, "forward")
    second = _background()
    _blob(second, 22, 24, 8, 8)
    _blob(second, 70, 90, 20, 20)  # a larger distractor well outside the gate
    obs = associator.update(second, 0.05, "forward", allow_reacquire=True)
    assert obs.status is DetectionStatus.DETECTED
    assert np.allclose(obs.reference, [27.5, 25.5])


def test_noise_robustness():
    rng = np.random.default_rng(7)
    image = _background()
    _blob(image, 40, 60, 24, 12)
    image += rng.normal(0.0, 0.05, image.shape)
    salt = rng.random(image.shape[:2]) < 0.01
    image[salt] = [0.9, 0.1, 0.1]
    image = np.clip(image, 0.0, 1.0)
    obs = detect_target(image)
    assert obs.status is DetectionStatus.DETECTED
    assert np.allclose(obs.reference, [65.5, 51.5], atol=1.0)


def test_integer_images_are_scaled_and_observations_are_frozen():
    image = (_background() * 255).astype(np.uint8)
    image[40:52, 60:72] = [217, 25, 25]
    obs = detect_target(image)
    assert obs.status is DetectionStatus.DETECTED
    with pytest.raises(FrozenInstanceError):
        obs.status = DetectionStatus.NO_FRAME
