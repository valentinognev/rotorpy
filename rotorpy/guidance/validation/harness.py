"""One closed-loop reliability trial (plan Phase 9): the real guidance stack driven
by a synthetic vehicle and camera instead of Gazebo/PX4.

`run_trial` builds the exact production `HomingGuidance` (`homing_config.
guidance_from_config`, the Gazebo/mounts path: tracking state machine, target-motion
estimator, command-limiting vehicle envelope, PN lead) and the real `SetpointStream`
(T3) it is handed over to, then steps a `vehicle_model.VehicleResponseModel` and
`target_world.MovingTargetWorldBuilder`/`PinholeCamera`s truth loop around it at a
fixed simulation rate, applying any `faults.FaultSchedule` windows. Termination comes
from the real truth-side `scoring.TruthScorer`, exactly as `camera_homing_tracker.py`
uses it; `run_lifecycle.RunLifecycle` guarantees exactly one `RunOutcome` per trial
even when a closer (for example recording) fails.

Nothing here feeds `HomingGuidance` anything but the vehicle's own simulated state
and rendered camera pixels: the target's true position reaches only the truth-side
scorer and the synthetic camera's world, never `guidance.update`'s arguments.
"""

import math
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np

from rotorpy.guidance.gazebo_bridge import attitude_homing_to_ned_frd
from rotorpy.guidance.homing import HomingCommand
from rotorpy.guidance.homing_config import (
    PROPOSED_REACH_RADIUS_M,
    guidance_from_config,
    vehicle_envelope,
)
from rotorpy.guidance.outcome import RunOutcome
from rotorpy.guidance.run_lifecycle import RunLifecycle
from rotorpy.guidance.scoring import TargetShape, TruthScorer
from rotorpy.guidance.setpoint_stream import SetpointStream
from rotorpy.guidance.step_decision import command_and_stamp
from rotorpy.guidance.tracking import TrackingState
from rotorpy.guidance.validation.faults import FaultSchedule
from rotorpy.guidance.validation.harness_config import harness_config
from rotorpy.guidance.validation.target_world import MovingTargetWorldBuilder, TargetMotionProfile
from rotorpy.guidance.validation.true_cameras import build_true_frd_cameras, world_to_true_ned
from rotorpy.guidance.validation.vehicle_model import VehicleResponseModel

# Proposed, harness-only: the wall-clock-equivalent delay between a transmitted
# command being issued and the vehicle actually starting to respond to it, and
# between a pose existing and the synthetic camera rendering it -- stand-ins for the
# real tracker's independent setpoint-publisher-thread cadence and MAVLink/ROS image
# pipeline latency, never measured on real hardware here.
PROPOSED_HARNESS_COMMAND_LATENCY_S = 0.05
PROPOSED_HARNESS_IMAGE_DELAY_S = 0.05
PROPOSED_HARNESS_SIM_DT_S = 0.02

_STATUS_TO_OUTCOME = {
    "ground": RunOutcome.GROUND_MISS,
    "timeout": RunOutcome.TIMEOUT,
    "target": RunOutcome.TARGET_REACHED,
    "target_lost": RunOutcome.TARGET_LOST,
}
_VISIBILITY_LOSS_STATES = (TrackingState.TEMPORARY_LOSS, TrackingState.REACQUIRE)


def _placeholder_command() -> HomingCommand:
    """A benign, unlocked zero command -- same shape as the real tracker's own
    `_placeholder_command()`, used only before the first real homing command and
    never built from target truth."""
    return HomingCommand(np.zeros(3), np.zeros(3), np.zeros(3), 0.0, False, False, None, None, None)


def _camera_due(t_s: float, period_s: float, tol_s: float = 1e-9) -> bool:
    """True at `t_s = 0` and within `tol_s` of a multiple of `period_s` (mirrors
    `runner._camera_due`'s float-tick convention)."""
    nearest = round(float(t_s) / period_s) * period_s
    return abs(float(t_s) - nearest) <= tol_s


def _is_finite_command(command: Optional[HomingCommand]) -> bool:
    if command is None:
        return True
    return bool(
        np.all(np.isfinite(command.velocity_ned))
        and np.all(np.isfinite(command.acceleration_ned))
        and np.isfinite(command.yaw_rad)
    )


@dataclass
class TrialSpec:
    """Everything one closed-loop trial needs besides the shared harness config.

    `on_tick`, if given, is called once per simulated tick with a `TickRecord` --
    release-gate tests use it to inspect the per-tick tracking state and the
    actually-transmitted command without re-implementing the loop themselves.
    """

    seed: int
    target_profile: TargetMotionProfile
    fault_schedule: FaultSchedule = field(default_factory=FaultSchedule)
    nav_constant: Optional[float] = None
    initial_heading_rad: Optional[float] = None
    initial_speed_m_s: Optional[float] = None
    sim_dt_s: float = PROPOSED_HARNESS_SIM_DT_S
    command_latency_s: float = PROPOSED_HARNESS_COMMAND_LATENCY_S
    image_delay_s: float = PROPOSED_HARNESS_IMAGE_DELAY_S
    max_ticks: Optional[int] = None
    on_tick: Optional[Any] = None


@dataclass
class TickRecord:
    """One simulated tick, for `TrialSpec.on_tick`."""

    t_s: float
    command: Optional[HomingCommand]
    transmitted: Optional[HomingCommand]
    tracking_state: Optional[TrackingState]


@dataclass
class TrialResult:
    """What one trial measured, reported together (plan Phase 9's release gate)."""

    outcome: RunOutcome
    reason: Optional[str]
    elapsed_s: float
    miss_m: Optional[float]
    mean_speed_m_s: float
    max_speed_m_s: float
    mean_tracking_error_m: Optional[float]
    visibility_loss_ticks: int
    visibility_loss_episodes: int
    mean_observation_age_s: Optional[float]
    max_observation_age_s: Optional[float]
    saturation_ticks: int
    ticks: int
    non_finite_detected: bool
    closer_failures: List[Any] = field(default_factory=list)


def _initial_course(config: dict, spec: TrialSpec):
    start_ned = np.asarray(config["start_ned"], dtype=float)
    target0 = np.asarray(spec.target_profile.position_ned(0.0), dtype=float)
    heading = spec.initial_heading_rad
    if heading is None:
        horizontal = target0[:2] - start_ned[:2]
        norm = float(np.linalg.norm(horizontal))
        heading = float(np.arctan2(horizontal[1], horizontal[0])) if norm > 1e-6 else 0.0
    speed = spec.initial_speed_m_s if spec.initial_speed_m_s is not None else float(config["initial_speed_m_s"])
    velocity_ned = speed * np.array([np.cos(heading), np.sin(heading), 0.0])
    return start_ned, velocity_ned, float(heading)


def run_trial(spec: TrialSpec, config: Optional[dict] = None) -> TrialResult:
    """Run one closed-loop trial to a defined `RunOutcome`. Deterministic for a
    given `(spec, config)`: no randomness is drawn during the loop itself, only (by
    convention) when a caller builds `spec` from `spec.seed`."""
    config = harness_config() if config is None else config
    if spec.nav_constant is not None:
        config = dict(config, nav_constant=float(spec.nav_constant))

    camera_period_s = 1.0 / float(config["camera_rate_hz"])
    timeout_s = float(config["timeout_s"])
    ground_down_m = -float(config["ground_z_m"])

    guidance = guidance_from_config(config)
    # True NED/FRD cameras (Task 10 fix), not `homing_config.gazebo_scene_from_config`'s
    # cameras: those assume the standard "simulator body" convention
    # (`scene.pitched_extrinsics`), which mismatches `attitude_sim_quat`'s actual
    # "homing simulator" (mirrored) convention once the vehicle yaws away from its
    # initial heading -- see `validation.true_cameras`'s module docstring and
    # `task-10-report.md`'s investigation log.
    forward_camera, upward_camera = build_true_frd_cameras(config)
    world_builder = MovingTargetWorldBuilder(config)

    start_ned, initial_velocity_ned, heading = _initial_course(config, spec)
    envelope = vehicle_envelope()
    vehicle = VehicleResponseModel(
        position_ned=start_ned,
        velocity_ned=initial_velocity_ned,
        yaw_rad=heading,
        tau_s=float(config["track_tau_s"]),
        max_accel_m_s2=float(envelope["max_accel_m_s2"]),
        max_tilt_deg=float(config["max_tilt_deg"]),
        min_specific_up_g=float(config["min_specific_up_g"]),
        max_yaw_rate_rad_s=float(envelope["max_yaw_rate_rad_s"]),
        command_latency_s=spec.command_latency_s,
        image_delay_s=spec.image_delay_s,
    )

    scorer = TruthScorer(
        TargetShape(float(config["gazebo_target_radius_m"]), float(config["gazebo_target_height_m"])),
        timeout_s=timeout_s,
        reach_radius_m=PROPOSED_REACH_RADIUS_M,
        ground_down_m=ground_down_m,
    )

    sim_clock = {"t": 0.0}
    transmitted_log: List[Optional[HomingCommand]] = []

    def _publish(command: Optional[HomingCommand]) -> None:
        transmitted_log.append(command)

    setpoints = SetpointStream(publish=_publish, now=lambda: sim_clock["t"])

    closer_failures: List[Any] = []

    def _recording_closer() -> None:
        if spec.fault_schedule.recording_failure:
            raise RuntimeError("synthetic recording failure (fault injection)")

    lifecycle = RunLifecycle(write_outcome=None, closers=[("recording", _recording_closer)])

    previous_command: Optional[HomingCommand] = None
    speeds: List[float] = []
    tracking_errors: List[float] = []
    observation_ages: List[float] = []
    visibility_loss_ticks = 0
    visibility_loss_episodes = 0
    was_losing = False
    saturation_ticks = 0
    non_finite_detected = False
    status: Optional[str] = None

    max_ticks = spec.max_ticks
    if max_ticks is None:
        max_ticks = int(math.ceil(timeout_s / spec.sim_dt_s)) + 10

    for tick in range(max_ticks):
        t = tick * spec.sim_dt_s
        sim_clock["t"] = t

        target_position_ned = np.asarray(spec.target_profile.position_ned(t), dtype=float)
        scorer.add_target_sample(t, target_position_ned)

        if spec.fault_schedule.pause_reset_due(t, spec.sim_dt_s / 2.0):
            guidance.reset_derivative_history()

        stalled = spec.fault_schedule.processing_stall_active(t)
        dropout = spec.fault_schedule.camera_dropout_active(t)
        forward_image = upward_image = None
        fresh = False
        capture_pose = None
        if not stalled and not dropout and _camera_due(t, camera_period_s):
            capture_pose = vehicle.delayed_pose(t)
            # True NED position / true FRD-in-NED attitude (Task 10 fix): matches
            # `build_true_frd_cameras`'s extrinsics convention and `homing.py`'s own
            # backprojection (`attitude_homing_to_ned_frd` is the same adapter
            # `_sightline_and_relative` uses), instead of the mismatched
            # "simulator body" pairing the previous `ned_to_sim(...)`/bare
            # `attitude_sim_quat` state used.
            state = {
                "x": capture_pose.position_ned,
                "q": attitude_homing_to_ned_frd(capture_pose.attitude_sim_quat),
            }
            world = world_to_true_ned(world_builder.world_at(target_position_ned))
            forward_image = forward_camera.measurement(state, world)["image"]
            upward_image = upward_camera.measurement(state, world)["image"]
            fresh = True

        have_estimate = not spec.fault_schedule.stale_telemetry_active(t)
        # Fix round 1 (Task 10, harness bug): on a fresh camera tick, the images were
        # rendered from `vehicle.delayed_pose(t)` (`image_delay_s` stale), so the
        # height/attitude/velocity handed to `guidance.update()` must describe that
        # SAME instant, not "now" -- `guidance.update()`'s own docstring requires
        # these three parameters to be "the vehicle state AT t [the measurement
        # time]", exactly mirroring how the real tracker samples them from the T3
        # `VehicleState` history at the image's own capture time. Feeding the
        # vehicle's live (not delayed) height/velocity into the same-tick ray
        # intersection and the target-motion estimator's Kalman-predict control
        # input was geometrically inconsistent with the delayed pixels and produced
        # a steadily growing spurious target-velocity estimate for even a genuinely
        # stationary target (see task-10-report.md's investigation log). When there
        # is no fresh frame this tick, there is no measurement to be consistent
        # with, so the vehicle's current state is used (matching `predict_to`'s own
        # "current, not historical" contract for the short processing-delay
        # extrapolation it performs).
        if capture_pose is not None:
            height_above_ground_m = ground_down_m - capture_pose.position_ned[2]
            attitude_for_update = capture_pose.attitude_sim_quat
            velocity_for_update = capture_pose.velocity_ned
        else:
            height_above_ground_m = ground_down_m - vehicle.position_ned[2]
            attitude_for_update = vehicle.attitude_sim_quat
            velocity_for_update = vehicle.velocity_ned

        def compute_command():
            return guidance.update(
                t, height_above_ground_m, attitude_for_update, velocity_for_update,
                forward_image, upward_image, control_t=t,
            )

        command, stamp_s = command_and_stamp(
            have_estimate=have_estimate, fresh=fresh, previous_command=previous_command,
            placeholder_command=_placeholder_command(), compute_command=compute_command,
            observation_wall_s=t,
        )
        previous_command = command
        if not _is_finite_command(command):
            non_finite_detected = True
        if stamp_s is not None:
            setpoints.set_command(command, stamp_s)

        if command is not None and getattr(command, "tracking_state", None) is TrackingState.ABORT and status is None:
            setpoints.mark_fatal("target_lost:" + (command.abort_reason or "unspecified"))
            status = "target_lost"

        losing = command is not None and getattr(command, "tracking_state", None) in _VISIBILITY_LOSS_STATES
        if losing:
            visibility_loss_ticks += 1
            if not was_losing:
                visibility_loss_episodes += 1
        was_losing = losing

        if command is not None and command.limiting_reasons:
            if any(not reason.startswith("target_motion_unavailable") for reason in command.limiting_reasons):
                saturation_ticks += 1

        transmitted = setpoints.tick(now_s=t)
        if not _is_finite_command(transmitted):
            non_finite_detected = True
        if transmitted is not None:
            vehicle.enqueue_command(t, transmitted.velocity_ned, transmitted.yaw_rad)

        if spec.on_tick is not None:
            spec.on_tick(TickRecord(
                t_s=t, command=command, transmitted=transmitted,
                tracking_state=None if command is None else getattr(command, "tracking_state", None),
            ))

        vehicle.step(spec.sim_dt_s, t + spec.sim_dt_s)
        speeds.append(float(np.linalg.norm(vehicle.velocity_ned)))
        if command is not None and command.tracking:
            tracking_errors.append(float(np.linalg.norm((vehicle.position_ned - target_position_ned)[:2])))
        age = setpoints.stats.last_observation_age_s
        if age is not None:
            observation_ages.append(float(age))

        own_status = scorer.add_vehicle_sample(t + spec.sim_dt_s, vehicle.position_ned)
        status = status or own_status
        if status is not None:
            break

    outcome = scorer.outcome(target_lost=(status == "target_lost"))
    if outcome is None:
        outcome = _STATUS_TO_OUTCOME.get(status, RunOutcome.SIMULATOR_FAILURE)
    lifecycle.finish(outcome, reason=status)
    closer_failures = lifecycle.close()

    result = scorer.result()
    return TrialResult(
        outcome=lifecycle.outcome,
        reason=status,
        elapsed_s=float(result.elapsed_s) if result.elapsed_s is not None else float(tick * spec.sim_dt_s),
        miss_m=scorer.miss_m(),
        mean_speed_m_s=float(np.mean(speeds)) if speeds else 0.0,
        max_speed_m_s=float(np.max(speeds)) if speeds else 0.0,
        mean_tracking_error_m=float(np.mean(tracking_errors)) if tracking_errors else None,
        visibility_loss_ticks=visibility_loss_ticks,
        visibility_loss_episodes=visibility_loss_episodes,
        mean_observation_age_s=float(np.mean(observation_ages)) if observation_ages else None,
        max_observation_age_s=float(np.max(observation_ages)) if observation_ages else None,
        saturation_ticks=saturation_ticks,
        ticks=tick + 1,
        non_finite_detected=non_finite_detected,
        closer_failures=closer_failures,
    )
