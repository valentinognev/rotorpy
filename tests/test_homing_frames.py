import numpy as np
from rotorpy.guidance.frames import frd_to_sim_body, ned_to_sim, sim_body_to_frd, sim_to_ned

def test_start_point_ned_to_sim():
    sim = ned_to_sim(np.array([0.0, 0.0, -30.0]))
    assert np.allclose(sim, [0.0, 0.0, 30.0])
    assert np.allclose(sim_to_ned(sim), [0.0, 0.0, -30.0])

def test_frd_flips_right_and_down():
    body = frd_to_sim_body(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(body, [1.0, -2.0, -3.0])
    assert np.allclose(sim_body_to_frd(body), [1.0, 2.0, 3.0])
