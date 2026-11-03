"""Synthetic-data unit tests for `Dockerfiles/px4_characterization.py`'s analysis
functions (plan Phase 7). `dump_px4_parameters`/`run_velocity_step_characterization`
need a live PX4/Gazebo mavlink link and are documented, not run, here -- only the
pure arithmetic is exercised, importable without `pymavlink`/`rospy`.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def characterization():
    spec = importlib.util.spec_from_file_location(
        "px4_characterization", ROOT / "Dockerfiles" / "px4_characterization.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _first_order_step(tau_s, step, duration_s=5.0, dt=0.05):
    times = np.arange(0.0, duration_s, dt)
    response = step * (1.0 - np.exp(-times / tau_s))
    commanded = np.full_like(times, step)
    return times, commanded, response


def test_first_order_step_response_rise_and_settling_match_the_known_time_constant(characterization):
    tau = 1.0
    times, commanded, response = _first_order_step(tau, step=5.0)
    metrics = characterization.analyze_step_response(times, commanded, response)
    # A first-order step's analytic 10%-90% rise time is tau * ln(9) ~= 2.197 tau.
    assert metrics.rise_time_s == pytest.approx(tau * np.log(9.0), abs=0.1)
    assert metrics.overshoot_frac == pytest.approx(0.0, abs=1e-9)
    assert metrics.settling_time_s is not None
    assert metrics.steady_state_error == pytest.approx(0.0, abs=0.05)


def test_overshoot_is_measured_relative_to_the_commanded_step(characterization):
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    commanded = np.full_like(times, 10.0)
    # Overshoots to 12 (20% over a step of 10), then settles exactly at 10.
    response = np.array([0.0, 4.0, 8.0, 11.0, 12.0, 11.0, 10.3, 10.0, 10.0, 10.0, 10.0])
    metrics = characterization.analyze_step_response(times, commanded, response)
    assert metrics.overshoot_frac == pytest.approx(0.2, abs=1e-9)


def test_settling_time_is_none_when_the_response_never_settles(characterization):
    times = np.linspace(0.0, 5.0, 50)
    commanded = np.full_like(times, 5.0)
    response = 5.0 + 2.0 * np.sin(times)  # keeps oscillating well outside the band
    metrics = characterization.analyze_step_response(times, commanded, response, settling_band_frac=0.01)
    assert metrics.settling_time_s is None


def test_zero_step_reports_steady_state_error_only(characterization):
    times = np.linspace(0.0, 1.0, 10)
    commanded = np.full_like(times, 3.0)
    response = np.full_like(times, 3.2)
    metrics = characterization.analyze_step_response(times, commanded, response)
    assert metrics.rise_time_s is None
    assert metrics.steady_state_error == pytest.approx(0.2, abs=1e-9)


def test_non_finite_input_is_rejected_without_raising(characterization):
    times = np.array([0.0, 1.0, np.nan])
    commanded = np.array([0.0, 5.0, 5.0])
    response = np.array([0.0, 4.0, 5.0])
    metrics = characterization.analyze_step_response(times, commanded, response)
    assert metrics.rise_time_s is None
    assert np.isnan(metrics.steady_state_error)


def test_analyze_velocity_tracking_reports_rms_and_max_error(characterization):
    commanded = np.array([1.0, 2.0, 3.0, 4.0])
    measured = np.array([1.1, 1.8, 3.3, 3.6])
    result = characterization.analyze_velocity_tracking(commanded, measured)
    errors = measured - commanded
    assert result["rms_error"] == pytest.approx(np.sqrt(np.mean(errors ** 2)))
    assert result["max_abs_error"] == pytest.approx(np.max(np.abs(errors)))


def test_relevant_px4_parameters_are_named_and_non_empty(characterization):
    assert len(characterization.RELEVANT_PX4_PARAMETERS) > 0
    assert all(isinstance(name, str) for name in characterization.RELEVANT_PX4_PARAMETERS)


def test_dump_px4_parameters_only_records_what_actually_answered(characterization):
    """Documented behavior, exercised with a fake mavlink connection (no live PX4):
    a name that never gets a matching `PARAM_VALUE` is simply absent, not a crash."""
    from types import SimpleNamespace

    answers = {"MPC_XY_VEL_MAX": 12.0}
    sent = []

    class FakeMav(object):
        def param_request_read_send(self, system, component, name, index):
            sent.append(name.decode("utf-8"))

    def recv_match(type=None, blocking=False):
        name = sent[-1] if sent else None
        if name in answers:
            value = answers.pop(name)
            return SimpleNamespace(param_id=name, param_value=value)
        return None

    connection = SimpleNamespace(
        mav=FakeMav(), target_system=1, target_component=1, recv_match=recv_match,
    )
    values = characterization.dump_px4_parameters(
        connection, names=("MPC_XY_VEL_MAX", "MPC_Z_VEL_MAX_UP"), timeout_s=0.05
    )
    assert values == {"MPC_XY_VEL_MAX": 12.0}
