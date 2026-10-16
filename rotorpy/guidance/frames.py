"""NED and body-FRD conversions into RotorPy simulator frames.

NED (north, east, down) maps to the simulator world (north, east, up) by
negating z. Body FRD (forward, right, down) maps to the simulator body
(forward, left, up) by negating y and z. Each conversion copies its input.
"""

import numpy as np


def ned_to_sim(ned: np.ndarray) -> np.ndarray:
    """Convert an NED vector of shape (3,) to the simulator world frame."""
    sim = np.array(ned, dtype=float, copy=True)
    sim[2] *= -1.0
    return sim


def sim_to_ned(sim: np.ndarray) -> np.ndarray:
    """Convert a simulator-world vector of shape (3,) to NED."""
    return ned_to_sim(sim)


def frd_to_sim_body(frd: np.ndarray) -> np.ndarray:
    """Convert a body-FRD vector of shape (3,) to the simulator body frame."""
    body = np.array(frd, dtype=float, copy=True)
    body[1] *= -1.0
    body[2] *= -1.0
    return body


def sim_body_to_frd(body: np.ndarray) -> np.ndarray:
    """Convert a simulator-body vector of shape (3,) to body FRD."""
    return frd_to_sim_body(body)
