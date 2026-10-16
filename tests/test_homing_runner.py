import numpy as np
from rotorpy.guidance.outcome import HomingOutcome
from rotorpy.guidance.runner import flight_status, format_nominal, format_sweep_line, preview_image

def test_stop_rules():
    assert flight_status(20.0, False, 30.0, np.zeros(3)) == "timeout"
    assert flight_status(19.99, False, 30.0, np.zeros(3)) is None
    assert flight_status(0.0, False, 0.0, np.zeros(3)) is None
    assert flight_status(5.0, True, 0.0, np.zeros(3)) == "ground"
    assert flight_status(5.0, True, 0.01, np.zeros(3)) is None
    assert flight_status(120.0, True, 10.0, np.zeros(3)) == "timeout"
    assert flight_status(119.0, True, 10.0, np.zeros(3)) is None
    assert flight_status(0.0, False, 30.0, np.array([20.0, 0.0, 0.0])) is None
    assert flight_status(0.0, False, 30.0, np.array([20.001, 0.0, 0.0])) == "timeout"
    assert flight_status(0.0, False, 30.0, np.array([0.0, 0.0, -20.0])) is None
    assert flight_status(0.0, False, 30.0, np.array([0.0, 0.0, -20.001])) == "timeout"
    assert flight_status(5.0, True, 0.0, np.array([25.0, 0.0, 0.0])) == "ground"
    assert flight_status(5.0, True, 0.0, np.array([0.0, 0.0, -25.0])) == "ground"

def test_preview_keeps_a_small_red_target():
    image = np.full((16, 16, 3), 0.9, dtype=np.float32)
    image[3, 5] = (1.0, 0.0, 0.0)
    preview = preview_image(image, side=4)
    assert preview.shape == (4, 4, 3)
    assert preview.dtype == np.uint8
    red = (
        (preview[:, :, 0] > 128)
        & (preview[:, :, 0] > preview[:, :, 1])
        & (preview[:, :, 0] > preview[:, :, 2])
    )
    assert np.count_nonzero(red) == 1


def test_print_lines():
    miss = HomingOutcome(miss_m=0.5, timed_out=False, time_s=11.0)
    timeout = HomingOutcome(miss_m=None, timed_out=True, time_s=20.0)
    assert format_nominal(miss) == "miss 0.500"
    assert format_nominal(timeout) == "timeout"
    assert format_sweep_line(2, miss) == "2 miss 0.500"
    assert format_sweep_line(4, timeout) == "4 timeout"
