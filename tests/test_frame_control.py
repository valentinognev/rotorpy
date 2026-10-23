import numpy as np
import pytest

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.frames import FrameConvention
from rotorpy.vehicles.crazyflie_params import quad_params


def test_frd_hover_command_is_identity_and_positive_thrust():
    ctrl = SE3Control(quad_params)
    ctrl.set_frame(FrameConvention.FRD_NED)
    ctrl.set_frame(FrameConvention.FRD_NED)
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0., 0., 0., 1.]), "w": np.zeros(3)}
    flat = {"x": np.zeros(3), "x_dot": np.zeros(3), "x_ddot": np.zeros(3), "yaw": 0.0, "yaw_dot": 0.0}
    cmd = ctrl.update(0.0, state, flat)
    assert cmd["cmd_thrust"] == pytest.approx(0.03 * 9.81, rel=1e-3)
    assert np.allclose(cmd["cmd_q"], [0, 0, 0, 1], atol=1e-6)
