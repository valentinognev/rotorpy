"""Closed-loop camera-homing flight. Images are rendered and discarded."""

import copy
from typing import List, Optional, Tuple

import numpy as np

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.guidance.camera_geometry import (
    pinhole_intrinsics,
    pitched_extrinsics,
    square_resolution,
)
from rotorpy.guidance.frames import sim_to_ned
from rotorpy.guidance.homing import HomingGuidance, flat_output
from rotorpy.guidance.outcome import HomingOutcome, miss_outcome, timeout_outcome
from rotorpy.guidance.scene import (
    INITIAL_SPEED,
    START_NED,
    TARGET_GROUND_NED,
    initial_velocity_ned,
    nominal_camera,
    nominal_initial_state,
    nominal_world,
)
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor

_SIM_DT = 0.01
_CAMERA_PERIOD = 0.05
_CAMERA_TOL = 1e-9


def flight_status(
    t: float,
    locked: bool,
    sim_z: float,
    sim_velocity: np.ndarray,
) -> Optional[str]:
    """Stop reason for the state about to be used, or None to continue.

    A locked sample with simulator z <= 0 is ground. That outranks over-speed.
    Otherwise timeout when any simulator velocity component has absolute value
    strictly greater than 20 m/s, when t >= 20 and the track is not locked, or
    when t >= 120 and the track is locked.
    """
    if locked and sim_z <= 0.0:
        return "ground"
    if np.any(np.abs(np.asarray(sim_velocity, dtype=float)) > 20):
        return "timeout"
    if t >= 20.0 and not locked:
        return "timeout"
    if t >= 120.0 and locked:
        return "timeout"
    return None


def _camera_due(t: float) -> bool:
    """True at t = 0 and within 1e-9 s of a multiple of 0.05 s."""
    nearest = round(float(t) / _CAMERA_PERIOD) * _CAMERA_PERIOD
    return abs(float(t) - nearest) <= _CAMERA_TOL


def _nav_label(nav_constant: float) -> str:
    value = float(nav_constant)
    if value.is_integer():
        return str(int(value))
    return "{:g}".format(value)


def run_homing(nav_constant: float = 3.0, lock_pixels: int = 36) -> HomingOutcome:
    """Fly the nominal scene until a locked ground contact or a timeout.

    Simulation is 100 Hz. The camera and homing law update at 20 Hz; other
    steps pass image=None. Wind stays zero. The flat-output position is the
    position before the step.
    """
    world = nominal_world()
    camera = nominal_camera()
    width, height = square_resolution()
    # Same 90 by 90 degree field of view and 10 degree pitch as nominal_camera.
    guidance = HomingGuidance(
        nav_constant=nav_constant,
        lock_pixels=lock_pixels,
        initial_velocity_ned=initial_velocity_ned(
            START_NED, TARGET_GROUND_NED, INITIAL_SPEED
        ),
        intrinsics=pinhole_intrinsics(width, height, 90.0, 90.0),
        extrinsics=pitched_extrinsics(10.0),
    )
    controller = SE3Control(quad_params)
    state = copy.deepcopy(nominal_initial_state())
    vehicle = Multirotor(
        quad_params,
        initial_state=state,
        control_abstraction="cmd_motor_speeds",
        aero=True,
        enable_ground=True,
    )

    locked = False
    step_index = 0
    while True:
        t = step_index * _SIM_DT
        status = flight_status(t, locked, float(state["x"][2]), state["v"])
        if status == "ground":
            return miss_outcome(sim_to_ned(state["x"]), t)
        if status == "timeout":
            return timeout_outcome(t)

        if _camera_due(t):
            image = camera.render(world, state)["image"]
        else:
            image = None
        command = guidance.update(
            t,
            float(state["x"][2]),
            state["q"],
            sim_to_ned(state["v"]),
            image,
        )
        if command.locked:
            locked = True

        flat = flat_output(state["x"], command)
        control = controller.update(t, state, flat)
        state = vehicle.step(state, control, _SIM_DT)
        state["wind"] = np.zeros(3)
        step_index += 1


def sweep(
    nav_constants: Tuple[float, ...] = (2, 3, 4, 5),
) -> List[Tuple[float, HomingOutcome]]:
    """One nominal flight per navigation constant, in the given order."""
    return [
        (nav_constant, run_homing(nav_constant=nav_constant))
        for nav_constant in nav_constants
    ]


def format_nominal(outcome: HomingOutcome) -> str:
    """`miss 0.500` with three decimal places, or `timeout`."""
    if outcome.timed_out:
        return "timeout"
    return "miss {:.3f}".format(outcome.miss_m)


def format_sweep_line(nav_constant: float, outcome: HomingOutcome) -> str:
    """`2 miss 0.500` or `2 timeout`."""
    return "{} {}".format(_nav_label(nav_constant), format_nominal(outcome))
