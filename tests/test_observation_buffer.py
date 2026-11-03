"""Per-camera bounded latest-observation buffering and explicit image decoding."""

import numpy as np
import pytest

from rotorpy.guidance.observation_buffer import CameraObservation, ObservationBuffer, decode_image, select_fresh_or_recent


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


def test_decode_does_not_require_trailing_padding_on_the_last_row():
    # A buffer with padded rows (step > tight row) commonly omits the padding bytes after
    # the very last row: only `(height - 1) * step + row_bytes` bytes are structurally
    # required, not a full `step * height` (which demands padding that may not exist).
    width, height, step = 4, 3, 20
    row_bytes = width * 3
    needed = (height - 1) * step + row_bytes
    data = _rgb_bytes(width, height, step=step)[:needed]
    image = decode_image(data, width=width, height=height, encoding="rgb8", step=step)
    assert image.shape == (height, width, 3)
    assert list(image[2, 3]) == [2, 3, 0]


def test_decode_still_rejects_a_buffer_missing_last_row_data():
    width, height, step = 4, 3, 20
    row_bytes = width * 3
    needed = (height - 1) * step + row_bytes
    data = _rgb_bytes(width, height, step=step)[: needed - 1]
    with pytest.raises(ValueError):
        decode_image(data, width=width, height=height, encoding="rgb8", step=step)


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


# --- select_fresh_or_recent --------------------------------------------------------


def _observation(receipt_s):
    return CameraObservation(capture_s=receipt_s, receipt_s=receipt_s, image=np.zeros((1, 1, 3)), encoding="rgb8")


def test_select_fresh_or_recent_returns_none_observations_as_none():
    assert select_fresh_or_recent(None, is_fresh=False, now_wall_s=1.0, max_age_s=1.0) is None
    assert select_fresh_or_recent(None, is_fresh=True, now_wall_s=1.0, max_age_s=1.0) is None


def test_select_fresh_or_recent_always_returns_a_genuinely_fresh_observation():
    obs = _observation(0.0)
    # Fresh this tick, regardless of its (irrelevant here) age.
    assert select_fresh_or_recent(obs, is_fresh=True, now_wall_s=100.0, max_age_s=0.1) is obs


def test_select_fresh_or_recent_keeps_a_stale_flagged_observation_within_max_age():
    # Not freshly offered this tick (the other camera's frame drove this tick), but still
    # within `max_age_s`: the still-valid cached observation is usable, with its own age.
    obs = _observation(1.0)
    assert select_fresh_or_recent(obs, is_fresh=False, now_wall_s=1.5, max_age_s=1.0) is obs


def test_select_fresh_or_recent_drops_an_observation_older_than_max_age():
    # A genuinely stale cached image (older than `max_age_s`) must not appear fresh/usable
    # just because it is the only thing in the slot.
    obs = _observation(1.0)
    assert select_fresh_or_recent(obs, is_fresh=False, now_wall_s=3.0, max_age_s=1.0) is None


def test_camera_slot_accounting_holds_under_concurrent_offer_and_take():
    # Every offered frame is exactly one of: taken fresh, dropped (overwritten unread), or
    # still pending at the end. Unsynchronised offer/take lose or double-count frames.
    import sys
    import threading

    from rotorpy.guidance.observation_buffer import CameraSlot

    slot = CameraSlot()
    image = np.zeros((1, 1, 3), dtype=np.uint8)
    offers_per_thread, offer_threads = 3000, 2
    taken = [0]
    done = threading.Event()
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def offerer():
        for k in range(offers_per_thread):
            slot.offer(float(k), float(k), image, "rgb8")

    def taker():
        while not done.is_set():
            _, was_fresh = slot.take_fresh()
            taken[0] += int(was_fresh)

    offerers = [threading.Thread(target=offerer) for _ in range(offer_threads)]
    reader = threading.Thread(target=taker)
    try:
        reader.start()
        for thread in offerers:
            thread.start()
        for thread in offerers:
            thread.join()
        done.set()
        reader.join()
    finally:
        sys.setswitchinterval(old_interval)
    _, pending = slot.take_fresh()
    assert taken[0] + slot.dropped + int(pending) == offers_per_thread * offer_threads
