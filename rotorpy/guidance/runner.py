"""Closed-loop camera-homing flight.

Guidance renders each frame and drops it. A recorded flight keeps
downsampled copies of both cameras for the example animation.
"""

import copy
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from rotorpy.sensors.camera import PinholeCamera
from rotorpy.world import World

from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.guidance.frames import sim_to_ned
from rotorpy.guidance.homing import HomingCommand, HomingGuidance, flat_output
from rotorpy.guidance.outcome import HomingOutcome, miss_outcome, timeout_outcome
from rotorpy.guidance.scene import (
    INITIAL_SPEED,
    START_NED,
    TARGET_GROUND_NED,
    initial_velocity_ned,
    nominal_camera,
    nominal_initial_state,
    nominal_upward_camera,
    nominal_world,
)
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor

_SIM_DT = 0.01
_CAMERA_PERIOD = 0.05
_CAMERA_TOL = 1e-9
_PREVIEW_SIDE = 280


@dataclass
class HomingRecord:
    """One homing flight, with camera frames small enough to animate."""

    time: np.ndarray
    position: np.ndarray
    quaternion: np.ndarray
    wind: np.ndarray
    image_time: np.ndarray
    forward_image: np.ndarray
    upward_image: np.ndarray
    source: List[Optional[str]]
    tracker_center: np.ndarray
    tracker_side: np.ndarray
    world: World
    forward_camera: PinholeCamera
    upward_camera: PinholeCamera
    preview_step: int


def preview_step(height: int, width: int, side: int = _PREVIEW_SIDE) -> int:
    """Integer block size `preview_image` uses for a frame of this size."""
    return max(1, int(np.ceil(max(height, width) / float(side))))


def preview_image(image: np.ndarray, side: int = _PREVIEW_SIDE) -> np.ndarray:
    """Shrink `image` so its long side is at most `side` pixels.

    Each output pixel is the reddest source pixel in its block. A plain
    average or nearest sample would hide the box while it still covers
    only a few pixels of the full frame.
    """
    frame = np.asarray(image)
    height, width = frame.shape[:2]
    step = preview_step(height, width, side)
    if step == 1:
        return _as_uint8(frame)
    cropped_h = (height // step) * step
    cropped_w = (width // step) * step
    # (block row, pixel row, block col, pixel col, channel)
    blocks = frame[:cropped_h, :cropped_w].reshape(
        cropped_h // step, step, cropped_w // step, step, frame.shape[2]
    )
    blocks = np.transpose(blocks, (0, 2, 1, 3, 4))
    flat = blocks.reshape(
        cropped_h // step, cropped_w // step, step * step, frame.shape[2]
    )
    redness = flat[..., 0] - np.maximum(flat[..., 1], flat[..., 2])
    pick = np.argmax(redness, axis=2)
    rows = np.arange(flat.shape[0])[:, None]
    cols = np.arange(flat.shape[1])[None, :]
    return _as_uint8(flat[rows, cols, pick])


def _as_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    return np.ascontiguousarray(np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8))


def flight_status(t: float, sim_z: float) -> Optional[str]:
    """Stop reason for the state about to be used, or None to continue.

    Ground when simulator z is at or below 0. That is checked first.
    Otherwise timeout when t >= 20 seconds. There is no speed stop.
    """
    if sim_z <= 0.0:
        return "ground"
    if t >= 20.0:
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
    """Fly the nominal scene until ground contact or a 20 second timeout.

    Simulation is 100 Hz. Both cameras and the homing law update at 20 Hz.
    The stopping sample is rendered and updated once, including when it is
    not a 20 Hz tick. Other steps pass both images as None. Wind stays zero.
    The flat-output position is the position before the step. Images are
    discarded.
    """
    outcome, _ = _fly_homing(nav_constant, lock_pixels, record=False)
    return outcome


def run_recorded_homing(
    nav_constant: float = 3.0, lock_pixels: int = 36
) -> Tuple[HomingOutcome, HomingRecord]:
    """Same flight as `run_homing`, plus the poses and both camera previews."""
    outcome, record = _fly_homing(nav_constant, lock_pixels, record=True)
    return outcome, record


def _tracker_sample(command: HomingCommand) -> Tuple[Optional[str], np.ndarray, float]:
    """One image time: source, or None, and NaN center and side when unset."""
    if command.tracker_center is None:
        center = np.array([np.nan, np.nan], dtype=float)
    else:
        center = np.array(command.tracker_center, dtype=float, copy=True)
    side = np.nan if command.tracker_side is None else float(command.tracker_side)
    return command.source, center, side


def _fly_homing(
    nav_constant: float, lock_pixels: int, record: bool
) -> Tuple[HomingOutcome, Optional[HomingRecord]]:
    world = nominal_world()
    forward_camera = nominal_camera()
    upward_camera = nominal_upward_camera()
    guidance = HomingGuidance(
        nav_constant=nav_constant,
        lock_pixels=lock_pixels,
        initial_velocity_ned=initial_velocity_ned(
            START_NED, TARGET_GROUND_NED, INITIAL_SPEED
        ),
        intrinsics=forward_camera.intrinsics,
        forward_extrinsics=forward_camera.extrinsics,
        upward_extrinsics=upward_camera.extrinsics,
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

    times: List[float] = []
    positions: List[np.ndarray] = []
    quaternions: List[np.ndarray] = []
    winds: List[np.ndarray] = []
    image_times: List[float] = []
    forward_images: List[np.ndarray] = []
    upward_images: List[np.ndarray] = []
    sources: List[Optional[str]] = []
    tracker_centers: List[np.ndarray] = []
    tracker_sides: List[float] = []

    def history() -> Optional[HomingRecord]:
        if not record:
            return None
        return HomingRecord(
            time=np.asarray(times, dtype=float),
            position=np.vstack(positions),
            quaternion=np.vstack(quaternions),
            wind=np.vstack(winds),
            image_time=np.asarray(image_times, dtype=float),
            forward_image=np.stack(forward_images),
            upward_image=np.stack(upward_images),
            source=list(sources),
            tracker_center=np.vstack(tracker_centers),
            tracker_side=np.asarray(tracker_sides, dtype=float),
            world=world,
            forward_camera=forward_camera,
            upward_camera=upward_camera,
            preview_step=preview_step(2236, 2236),
        )

    step_index = 0
    while True:
        t = step_index * _SIM_DT
        sim_z = float(state["x"][2])
        status = flight_status(t, sim_z)
        if record:
            times.append(t)
            positions.append(np.array(state["x"], dtype=float, copy=True))
            quaternions.append(np.array(state["q"], dtype=float, copy=True))
            winds.append(np.array(state["wind"], dtype=float, copy=True))

        stopping = status is not None
        if stopping or _camera_due(t):
            forward_image = forward_camera.render(world, state)["image"]
            upward_image = upward_camera.render(world, state)["image"]
        else:
            forward_image = None
            upward_image = None

        command = guidance.update(
            t,
            sim_z,
            state["q"],
            sim_to_ned(state["v"]),
            forward_image,
            upward_image,
        )
        if record and forward_image is not None:
            source, center, side = _tracker_sample(command)
            forward_images.append(preview_image(forward_image))
            upward_images.append(preview_image(upward_image))
            image_times.append(t)
            sources.append(source)
            tracker_centers.append(center)
            tracker_sides.append(side)

        if status == "ground":
            return miss_outcome(sim_to_ned(state["x"]), t), history()
        if status == "timeout":
            return timeout_outcome(t), history()

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
