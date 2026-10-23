import numpy as np
import torch

from rotorpy.frames import FrameConvention
from rotorpy.sensors.imu import BatchedImu, Imu


def test_frd_specific_force_at_rest_is_minus_z():
    imu = Imu(p_BS=np.zeros(3), R_BS=np.eye(3), sampling_rate=100)
    imu.set_frame(FrameConvention.FRD_NED)
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0., 0., 0., 1.]), "w": np.zeros(3)}
    statedot = {"vdot": np.zeros(3), "wdot": np.zeros(3)}
    meas = imu.measurement(state, statedot, with_noise=False)
    assert np.allclose(meas["accel"], [0, 0, -9.81], atol=1e-6)


def _rest_state():
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0., 0., 0., 1.]), "w": np.zeros(3)}
    statedot = {"vdot": np.zeros(3), "wdot": np.zeros(3)}
    return state, statedot


def test_frd_accel_noise_contribution_is_body_y_and_z_negated(monkeypatch):
    """A missing M_FLU_TO_FRD on the accel draw leaves the contribution at [0, 1, 2]."""
    monkeypatch.setattr(np.random, "normal", lambda *args, **kwargs: np.array([0.0, 1.0, 2.0]))

    def measure(with_noise):
        # sampling_rate=2 makes rate_scale = sqrt(sampling_rate/2) = 1, so the
        # added sample is the draw itself.
        imu = Imu(p_BS=np.zeros(3), R_BS=np.eye(3), sampling_rate=2)
        imu.set_frame(FrameConvention.FRD_NED)
        imu.set_frame(FrameConvention.FRD_NED)
        state, statedot = _rest_state()
        return imu.measurement(state, statedot, with_noise=with_noise)

    quiet = measure(False)
    noisy = measure(True)
    # Bias walk of [0, 1, 2] mapped by diag(1, -1, -1) is [0, -1, -2],
    # added to specific force [0, 0, -9.81].
    assert np.allclose(quiet["accel"], [0.0, -1.0, -11.81], atol=1e-6)
    assert np.allclose(noisy["accel"] - quiet["accel"], [0.0, -1.0, -2.0], atol=1e-6)


def test_batched_imu_frd_gravity_vector_is_plus_z():
    imu = BatchedImu(1, device="cpu")
    imu.set_frame(FrameConvention.FRD_NED)
    imu.set_frame(FrameConvention.FRD_NED)
    assert tuple(imu.gravity_vector.shape) == (3,)
    assert imu.gravity_vector.device.type == "cpu"
    assert torch.allclose(
        imu.gravity_vector,
        torch.tensor([0.0, 0.0, 9.81], dtype=torch.double),
    )


def test_batched_frd_noise_and_bias_walk_negate_body_y_and_z(monkeypatch):
    """Draws of [0, 1, 2] must be multiplied by diag(1, -1, -1) before they are added."""

    def fake_randn_like(tensor, *args, **kwargs):
        draw = torch.tensor([0.0, 1.0, 2.0], device=tensor.device, dtype=tensor.dtype)
        return draw.expand(tensor.shape).clone()

    monkeypatch.setattr(torch, "randn_like", fake_randn_like)
    ones = torch.ones(3)
    zeros = torch.zeros(3)
    params = {
        "initial_bias": zeros,
        "noise_density": ones,
        "random_walk": ones,
    }
    imu = BatchedImu(
        1,
        accelerometer_params=params,
        gyroscope_params=dict(params),
        sampling_rate=2,
        device="cpu",
    )
    imu.set_frame(FrameConvention.FRD_NED)
    state = {
        "x": torch.zeros(1, 3).double(),
        "v": torch.zeros(1, 3).double(),
        "q": torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.double),
        "w": torch.zeros(1, 3).double(),
    }
    statedot = {
        "vdot": torch.zeros(1, 3).double(),
        "wdot": torch.zeros(1, 3).double(),
    }
    quiet = imu.measurement(state, statedot, with_noise=False)
    noisy_imu = BatchedImu(
        1,
        accelerometer_params=params,
        gyroscope_params=dict(params),
        sampling_rate=2,
        device="cpu",
    )
    noisy_imu.set_frame(FrameConvention.FRD_NED)
    noisy = noisy_imu.measurement(state, statedot, with_noise=True)
    assert torch.allclose(
        quiet["accel"][0],
        torch.tensor([0.0, -1.0, -11.81], dtype=torch.double),
        atol=1e-6,
    )
    assert torch.allclose(
        noisy["accel"][0] - quiet["accel"][0],
        torch.tensor([0.0, -1.0, -2.0], dtype=torch.double),
        atol=1e-6,
    )
