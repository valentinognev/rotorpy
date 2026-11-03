"""Per-camera bounded latest-observation buffering and explicit image decoding."""

import numpy as np
import pytest

from rotorpy.guidance.observation_buffer import ObservationBuffer, decode_image


def _rgb_bytes(width, height, step=None):
    """height*step bytes where pixel (row, col) is (row, col, 0) in the first 3 channels."""
    channels = 3
    row_bytes = width * channels
    step = row_bytes if step is None else step
    buf = np.zeros((height, step), dtype=np.uint8)
    for row in range(height):
        for col in range(width):
            buf[row, col * channels:col * channels + channels] = [row, col, 0]
    return buf.tobytes()


def test_decode_rgb8_round_trips_pixel_values():
    data = _rgb_bytes(4, 3)
    image = decode_image(data, width=4, height=3, encoding="rgb8", step=4 * 3)
    assert image.shape == (3, 4, 3)
    assert image.dtype == np.uint8
    assert list(image[2, 1]) == [2, 1, 0]


def test_decode_bgr8_reorders_to_rgb():
    channels = 3
    width, height = 2, 2
    buf = np.zeros((height, width * channels), dtype=np.uint8)
    buf[0, 0:3] = [10, 20, 30]  # stored as B, G, R
    image = decode_image(buf.tobytes(), width=width, height=height, encoding="bgr8", step=width * channels)
    assert list(image[0, 0]) == [30, 20, 10]


def test_decode_bgra8_reorders_and_drops_alpha():
    width, height = 1, 1
    buf = np.array([[40, 50, 60, 255]], dtype=np.uint8)  # B, G, R, A
    image = decode_image(buf.tobytes(), width=width, height=height, encoding="bgra8", step=4)
    assert image.shape == (1, 1, 3)
    assert list(image[0, 0]) == [60, 50, 40]


def test_decode_rgba8_drops_alpha_without_reordering():
    width, height = 1, 1
    buf = np.array([[60, 50, 40, 255]], dtype=np.uint8)
    image = decode_image(buf.tobytes(), width=width, height=height, encoding="rgba8", step=4)
    assert list(image[0, 0]) == [60, 50, 40]


def test_decode_mono8_produces_equal_three_channels():
    width, height = 2, 1
    buf = np.array([[7, 200]], dtype=np.uint8)
    image = decode_image(buf.tobytes(), width=width, height=height, encoding="mono8", step=2)
    assert image.shape == (1, 2, 3)
    assert list(image[0, 0]) == [7, 7, 7]
    assert list(image[0, 1]) == [200, 200, 200]


def test_decode_honours_row_stride_padding():
    # step is larger than the tight row (width * channels): padding bytes must be skipped.
    data = _rgb_bytes(4, 3, step=20)
    image = decode_image(data, width=4, height=3, encoding="rgb8", step=20)
    assert image.shape == (3, 4, 3)
    assert list(image[2, 3]) == [2, 3, 0]


def test_decode_rejects_unsupported_encoding():
    with pytest.raises(ValueError, match="encoding"):
        decode_image(b"\x00" * 100, width=4, height=4, encoding="yuv422", step=16)


def test_decode_rejects_buffer_too_small_for_stride_and_height():
    with pytest.raises(ValueError):
        decode_image(b"\x00" * 4, width=4, height=4, encoding="rgb8", step=12)


def test_decode_rejects_stride_smaller_than_one_tight_row():
    with pytest.raises(ValueError):
        decode_image(b"\x00" * 100, width=4, height=4, encoding="rgb8", step=8)


def test_buffer_slots_are_independent_per_camera():
    buffer = ObservationBuffer(cameras=("forward", "upward"))
    data = _rgb_bytes(2, 2)
    buffer.offer("forward", capture_s=1.0, receipt_s=1.1, raw_bytes=data, width=2, height=2,
                 encoding="rgb8", step=2 * 3)
    forward_obs, forward_fresh = buffer.take_fresh("forward")
    upward_obs, upward_fresh = buffer.take_fresh("upward")
    assert forward_fresh is True
    assert forward_obs is not None
    # Only the forward camera received anything: the upward slot stays empty, not "fresh".
    assert upward_fresh is False
    assert upward_obs is None


def test_take_fresh_is_true_only_once_per_offer():
    buffer = ObservationBuffer(cameras=("forward",))
    data = _rgb_bytes(2, 2)
    buffer.offer("forward", capture_s=1.0, receipt_s=1.1, raw_bytes=data, width=2, height=2,
                 encoding="rgb8", step=2 * 3)
    _, first = buffer.take_fresh("forward")
    observation, second = buffer.take_fresh("forward")
    assert first is True
    assert second is False
    # The latest observation is still readable even once it is no longer "fresh".
    assert observation is not None


def test_overwriting_an_unread_observation_increments_the_drop_counter():
    buffer = ObservationBuffer(cameras=("forward",))
    data = _rgb_bytes(2, 2)
    buffer.offer("forward", capture_s=1.0, receipt_s=1.1, raw_bytes=data, width=2, height=2,
                 encoding="rgb8", step=2 * 3)
    assert buffer.dropped("forward") == 0
    buffer.offer("forward", capture_s=2.0, receipt_s=2.1, raw_bytes=data, width=2, height=2,
                 encoding="rgb8", step=2 * 3)
    assert buffer.dropped("forward") == 1
    observation, fresh = buffer.take_fresh("forward")
    assert fresh is True
    assert observation.capture_s == 2.0


def test_offer_records_capture_and_receipt_times():
    buffer = ObservationBuffer(cameras=("forward",))
    data = _rgb_bytes(2, 2)
    buffer.offer("forward", capture_s=5.0, receipt_s=5.2, raw_bytes=data, width=2, height=2,
                 encoding="rgb8", step=2 * 3)
    observation, _ = buffer.take_fresh("forward")
    assert observation.capture_s == 5.0
    assert observation.receipt_s == 5.2
    assert observation.encoding == "rgb8"
    assert observation.image.shape == (2, 2, 3)


def test_unknown_camera_name_raises_key_error():
    buffer = ObservationBuffer(cameras=("forward",))
    with pytest.raises(KeyError):
        buffer.take_fresh("upward")
