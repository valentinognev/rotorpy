"""
Camera homing.

Prints one nominal miss or timeout, then opens a third-person 3D view of
the flight beside the drone's camera image. With --sweep, prints one line
each for navigation constants 2, 3, 4, and 5.
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rotorpy.guidance.runner import (
    HomingRecord,
    format_nominal,
    format_sweep_line,
    run_recorded_homing,
    sweep,
)
from rotorpy.utils.animate import animate


def _scenario_extents(position: np.ndarray, world, margin: float = 15.0) -> list:
    """Cube around the flown path and the target box, in simulator meters."""
    block = world.world["blocks"][0]["extents"]
    corners = np.array([
        [block[0], block[2], block[4]],
        [block[1], block[3], block[5]],
    ])
    points = np.vstack([np.asarray(position, dtype=float), corners])
    low = points.min(axis=0) - margin
    high = points.max(axis=0) + margin
    span = float(np.max(high - low))
    center = 0.5 * (low + high)
    half = 0.5 * span
    low = center - half
    high = center + half
    return [low[0], high[0], low[1], high[1], low[2], high[2]]


def show_homing(record: HomingRecord):
    """Play the flight the way camera_demo does: 3D scene beside the camera."""
    record.world.world["bounds"]["extents"] = _scenario_extents(
        record.position, record.world
    )
    rotation = Rotation.from_quat(record.quaternion).as_matrix()
    animation = animate(
        record.time,
        record.position,
        rotation,
        record.wind,
        False,
        record.world,
        camera_images=record.image,
        camera_times=record.image_time,
        camera=record.camera,
    )
    ax = plt.figure("Animation").axes[0]
    ax.plot(
        record.position[:, 0],
        record.position[:, 1],
        record.position[:, 2],
        color="tab:blue",
        linewidth=1.5,
    )
    plt.show()
    return animation


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hummingbird camera-homing flight.")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Sweep navigation constants 2, 3, 4, and 5.",
    )
    args = parser.parse_args()
    if args.sweep:
        for nav_constant, outcome in sweep((2, 3, 4, 5)):
            print(format_sweep_line(nav_constant, outcome))
    else:
        outcome, record = run_recorded_homing()
        print(format_nominal(outcome), flush=True)
        show_homing(record)
