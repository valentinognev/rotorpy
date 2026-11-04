import pathlib
import pytest
from rotorpy.guidance.homing_config import load_homing_config
from rotorpy.guidance.runner import format_sweep_line, sweep, run_homing

def test_default_suite_skips_homing_e2e():
    text = pathlib.Path("pyproject.toml").read_text()
    assert "not homing_e2e" in text
    assert "homing_e2e:" in text

@pytest.mark.homing_e2e
def test_never_lock_times_out_at_20_seconds():
    outcome = run_homing(lock_pixels=10**9)
    assert outcome.timed_out is True
    assert outcome.miss_m is None
    assert outcome.time_s == pytest.approx(20.0, abs=0.011)

@pytest.mark.homing_e2e
@pytest.mark.parametrize("mode", ["attitude", "rate"])
def test_angle_modes_home_as_well_as_velocity(mode):
    base = load_homing_config("examples/camera_homing.jsonc") | {"controller": "pid"}
    velocity = run_homing(config=base | {"command_mode": "velocity"})
    angle = run_homing(config=base | {"command_mode": mode})
    print("velocity", velocity, mode, angle)
    assert not velocity.timed_out
    assert not angle.timed_out
    assert angle.miss_m <= max(2.0, (velocity.miss_m or 0.0) + 1.0)

@pytest.mark.homing_e2e
def test_sweep_prints_one_outcome_per_constant(capsys):
    results = sweep((2, 3, 4, 5))
    assert [n for n, _ in results] == [2, 3, 4, 5]
    for nav_constant, outcome in results:
        assert outcome.timed_out or outcome.miss_m >= 0.0
        print(format_sweep_line(nav_constant, outcome))
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert lines == [format_sweep_line(n, outcome) for n, outcome in results]
