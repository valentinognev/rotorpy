import numpy as np
from rotorpy.guidance.outcome import miss_outcome, timeout_outcome

def test_ground_point_is_half_a_meter_from_the_volume_center():
    outcome = miss_outcome(np.array([100.0, 50.0, 0.0]), 12.0)
    assert outcome.miss_m == 0.5
    assert outcome.timed_out is False
    assert outcome.time_s == 12.0

def test_timeout_has_no_miss():
    outcome = timeout_outcome(20.0)
    assert outcome.miss_m is None
    assert outcome.timed_out is True
    assert outcome.time_s == 20.0
