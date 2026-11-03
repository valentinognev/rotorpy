"""Held-out false-positive / missed-detection evaluation harness for `detect_target`.

All frames here are synthetic. Rates reported in these tests describe this synthetic
labeled set only -- they are not a claim about real Gazebo-recorded performance.
"""

import numpy as np

from rotorpy.guidance.pixels import (
    DEFAULT_CONFIG,
    DetectionRates,
    LabeledFrame,
    evaluate_detection_rates,
    split_tuning_and_held_out,
)

RED = [0.85, 0.1, 0.1]
WEAK_RED = [0.6, 0.35, 0.35]


def _background(height=96, width=128, base=(0.35, 0.5, 0.6)):
    image = np.empty((height, width, 3))
    image[:] = base
    return image


def _blob(image, row, col, height, width, color=RED):
    image[row:row + height, col:col + width] = color
    return image


def _textured(seed, height=96, width=128):
    rng = np.random.default_rng(seed)
    image = _background(height, width)
    image += rng.normal(0.0, 0.03, image.shape)
    return np.clip(image, 0.0, 1.0)


def _shadowed(seed, height=96, width=128):
    image = _textured(seed, height, width)
    image[:40, :] *= 0.4  # a dark band across the top, away from the target placements below
    return image


def _lit(seed, height=96, width=128, brightness=1.0):
    image = _textured(seed, height, width) * brightness
    return np.clip(image, 0.0, 1.0)


def _positive(image, row, col, height, width, color=RED):
    _blob(image, row, col, height, width, color)
    reference = (col + width / 2.0 - 0.5, row + height / 2.0 - 0.5)
    return LabeledFrame(image=image, has_target=True, reference=reference)


def _negative(image):
    return LabeledFrame(image=image, has_target=False)


def _labeled_frames():
    frames = []
    # Lighting changes: same target, darker/brighter backgrounds.
    for seed, brightness in enumerate([0.6, 0.85, 1.0, 1.2]):
        frames.append(_positive(_lit(seed, brightness=brightness), 40, 60, 12, 12))
    # Textured backgrounds, no shadows.
    for seed in range(4, 8):
        frames.append(_positive(_textured(seed), 30, 70, 10, 10))
    # Shadows overlapping only the background, not the target.
    for seed in range(8, 11):
        frames.append(_positive(_shadowed(seed), 60, 50, 10, 16))
    # Varying distance, including tiny distant targets that must still be detected.
    for seed, size in enumerate([3, 4, 6, 10, 18], start=11):
        frames.append(_positive(_textured(seed), 50, 20 + 4 * size, size, size))
    # Red distractors: a weaker/smaller distractor elsewhere must not beat the real target.
    for seed in range(16, 19):
        image = _textured(seed)
        _blob(image, 10, 100, 6, 6, WEAK_RED)
        frames.append(_positive(image, 55, 40, 14, 14))
    # Negative frames: no target anywhere, across the same background categories.
    for seed in range(19, 23):
        frames.append(_negative(_textured(seed)))
    for seed in range(23, 26):
        frames.append(_negative(_shadowed(seed)))
    return frames


def test_split_keeps_tuning_and_held_out_disjoint_and_covers_every_frame():
    frames = _labeled_frames()
    tuning, held_out = split_tuning_and_held_out(frames, held_out_fraction=0.5, rng=np.random.default_rng(0))
    assert len(tuning) + len(held_out) == len(frames)
    assert not (set(id(f) for f in tuning) & set(id(f) for f in held_out))


def test_split_is_deterministic_for_a_given_rng_seed():
    frames = _labeled_frames()
    tuning_a, held_out_a = split_tuning_and_held_out(frames, held_out_fraction=0.4, rng=np.random.default_rng(3))
    tuning_b, held_out_b = split_tuning_and_held_out(frames, held_out_fraction=0.4, rng=np.random.default_rng(3))
    assert [id(f) for f in tuning_a] == [id(f) for f in tuning_b]
    assert [id(f) for f in held_out_a] == [id(f) for f in held_out_b]


def test_detection_rates_on_held_out_synthetic_frames():
    frames = _labeled_frames()
    tuning, held_out = split_tuning_and_held_out(frames, held_out_fraction=0.5, rng=np.random.default_rng(1))
    assert len(held_out) > 0

    rates = evaluate_detection_rates(held_out, config=DEFAULT_CONFIG)
    assert isinstance(rates, DetectionRates)
    assert rates.num_frames == len(held_out)
    assert rates.num_positive + rates.num_negative == len(held_out)
    # Synthetic-set rates, not a verified real-world claim.
    assert rates.missed_detection_rate <= 0.2
    assert rates.false_positive_rate <= 0.2


def test_missed_detection_rate_is_one_when_every_positive_frame_is_undetected():
    frames = [_positive(_background(), 40, 60, 1, 1, WEAK_RED) for _ in range(3)]
    rates = evaluate_detection_rates(frames)
    assert rates.num_positive == 3
    assert rates.missed_detection_rate == 1.0
    assert rates.false_positive_rate == 0.0


def test_false_positive_rate_is_one_when_every_negative_frame_is_detected():
    frames = [_negative(_blob(_background(), 40, 60, 10, 10)) for _ in range(3)]
    rates = evaluate_detection_rates(frames)
    assert rates.num_negative == 3
    assert rates.false_positive_rate == 1.0
    assert rates.missed_detection_rate == 0.0


def test_a_detection_far_from_the_labeled_reference_counts_as_a_miss():
    image = _background()
    _blob(image, 40, 60, 10, 10)
    far_reference = (0.0, 0.0)
    frame = LabeledFrame(image=image, has_target=True, reference=far_reference)
    rates = evaluate_detection_rates([frame], match_radius_px=5.0)
    assert rates.missed_detection_rate == 1.0
