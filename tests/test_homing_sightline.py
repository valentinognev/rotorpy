import numpy as np
from rotorpy.guidance.sightline import ground_relative_position

def test_straight_down_range_is_the_height():
    relative = ground_relative_position(30.0, np.array([0.0, 0.0, 2.0]))
    assert np.allclose(relative, [0.0, 0.0, 30.0])

def test_scaled_sightline():
    relative = ground_relative_position(30.0, np.array([3.0, 0.0, 4.0]))
    assert np.allclose(relative, [22.5, 0.0, 30.0])

def test_level_or_upward_sightline_is_lost():
    assert ground_relative_position(30.0, np.array([1.0, 0.0, 0.0])) is None
    assert ground_relative_position(30.0, np.array([1.0, 0.0, -0.1])) is None
    assert ground_relative_position(0.0, np.array([0.0, 0.0, 1.0])) is None
