"""
Camera homing.

Prints one nominal miss or timeout, then writes a Plotly page of the flight
to Results/camera_homing.html. The page is a 3D scene beside the forward and
upward pictures, with both camera frustums and the tracking square, and a
slider for the capture time. With --sweep, prints one line
each for navigation constants 2, 3, 4, and 5.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rotorpy.guidance.homing_page import write_homing_page
from rotorpy.guidance.runner import (
    HomingRecord,
    format_nominal,
    format_sweep_line,
    run_recorded_homing,
    sweep,
)


def show_homing(record: HomingRecord, path=None):
    """Write the flight to a Plotly page and return that path."""
    if path is None:
        path = os.path.abspath(os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "Results", "camera_homing.html"
        ))
    return write_homing_page(record, path)


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
        print(show_homing(record), flush=True)
