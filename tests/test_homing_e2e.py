import pathlib
import pytest
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
def test_sweep_prints_one_outcome_per_constant(capsys):
    results = sweep((2, 3, 4, 5))
    assert [n for n, _ in results] == [2, 3, 4, 5]
    for nav_constant, outcome in results:
        assert outcome.timed_out or outcome.miss_m >= 0.0
        print(format_sweep_line(nav_constant, outcome))
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert lines == [format_sweep_line(n, outcome) for n, outcome in results]
