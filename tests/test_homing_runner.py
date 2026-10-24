import inspect
import math
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from rotorpy.controllers.px4_cascade import Px4Cascade
from rotorpy.frames import attitude_enu_flu_to_ned_frd, enu_to_ned
from rotorpy.guidance.homing import HomingCommand
from rotorpy.guidance.outcome import HomingOutcome
from rotorpy.guidance.runner import (
    _fly_homing,
    flight_status,
    format_nominal,
    format_sweep_line,
    preview_image,
    preview_step,
    px4_command_from_homing,
)
from rotorpy.guidance.scene import (
    INITIAL_SPEED,
    START_NED,
    TARGET_GROUND_NED,
    initial_velocity_ned,
    nominal_initial_state,
)
from rotorpy.vehicles.hummingbird_params import quad_params

def test_stop_rules():
    assert flight_status(0.0, 0.0) == "ground"
    assert flight_status(5.0, 0.0) == "ground"
    assert flight_status(20.0, 0.0) == "ground"
    assert flight_status(20.0, 10.0) == "timeout"
    assert flight_status(19.99, 10.0) is None
    assert flight_status(0.0, 30.0) is None
    assert preview_step(2236, 2236) == 8

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


def test_homing_command_matches_px4_ned_at_the_no_lock_start():
    state = nominal_initial_state()
    homing_yaw = float(np.arctan2(TARGET_GROUND_NED[1], TARGET_GROUND_NED[0]))
    command = HomingCommand(
        initial_velocity_ned(START_NED, TARGET_GROUND_NED, INITIAL_SPEED),
        np.zeros(3),
        np.zeros(3),
        homing_yaw,
        False,
        False,
        None,
        None,
        None,
    )
    px4 = px4_command_from_homing(command)
    assert px4["tier"] == "velocity"
    assert px4["yaw_rate"] == 0.0
    assert np.allclose(px4["velocity_ned"], enu_to_ned(state["v"]))
    assert np.allclose(px4["acceleration_ned"], 0.0)
    ned_yaw = float(Rotation.from_quat(attitude_enu_flu_to_ned_frd(state["q"])).as_euler("ZYX")[0])
    assert px4["yaw"] == pytest.approx(ned_yaw)
    assert px4["yaw"] == pytest.approx(math.pi / 2.0 - homing_yaw)
    shifted = px4_command_from_homing(HomingCommand(
        np.array([1.0, 2.0, 3.0]),
        np.array([4.0, 5.0, 6.0]),
        np.zeros(3),
        0.0,
        False,
        False,
        None,
        None,
        None,
    ))
    assert np.allclose(shifted["velocity_ned"], [2.0, 1.0, 3.0])
    assert np.allclose(shifted["acceleration_ned"], [5.0, 4.0, 6.0])
    controller = Px4Cascade(quad_params)
    out = controller.update(0.0, state, px4)
    assert np.allclose(out["cmd_v"], state["v"], atol=1e-8)
    assert np.linalg.norm(out["cmd_acc"][:2]) < 1e-6
    source = inspect.getsource(_fly_homing)
    assert "make_controller" in source
    assert "control_from_homing" in source
    assert "set_frame" not in source
