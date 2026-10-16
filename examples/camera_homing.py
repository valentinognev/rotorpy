"""
Camera homing.

Prints one nominal miss or timeout. With --sweep, prints one line each for
navigation constants 2, 3, 4, and 5. No plots and no animation.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from rotorpy.guidance.runner import format_nominal, format_sweep_line, run_homing, sweep


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
        print(format_nominal(run_homing()))
