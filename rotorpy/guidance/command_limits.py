"""Vehicle-envelope and continuity limiting for one coherent guidance command
(plan Phase 7).

`limit_command` is the single place a guidance command's velocity, acceleration, and
yaw are bounded before leaving `HomingGuidance`: total speed (norm, not per-component --
a maximum-feasible-speed objective inside the vehicle envelope), vertical rate, raw
acceleration magnitude, commanded-acceleration (continuity) and commanded-jerk steps
per tick, yaw rate, and a visibility margin that scales the speed cap down as the
tracked target nears the image edge (speed/turn authority is reduced to protect the
observation, never to produce a terminal hover or distance-based stop). Every bound
that actually did something appends a named reason, so a caller's telemetry can record
why a command differs from what guidance originally asked for (T1 telemetry schema's
`requested_setpoint`/`limited_setpoint`/`limiting_reasons`).

Non-finite input (NaN/inf anywhere in the request, or a non-finite `dt`) is rejected
outright rather than silently propagated: the whole call falls back to the previous
(already-limited, already-finite) command, or to zero/identity if there is no
previous command yet (first tick) -- a command must never leave this function
non-finite. All limits are named `VehicleEnvelope` fields, documented at the config
layer (`homing_config.vehicle_envelope()`) as proposed, not measured.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class VehicleEnvelope:
    """Proposed vehicle limits. Every field is a required argument (mirrors
    `tracking.TrackingStateMachine`/`target_estimator.TargetMotionEstimator`): the
    actual proposed numbers are named in `homing_config.vehicle_envelope()`."""

    max_speed_m_s: float
    max_accel_m_s2: float
    max_jerk_m_s3: float
    max_yaw_rate_rad_s: float
    max_vertical_speed_m_s: float
    # Fraction of the half-extent (along the nearer image axis) inside which the
    # target counts as "near the edge" and the speed cap scales down.
    visibility_margin_frac: float = 0.2
    # Speed-cap scale at the image border itself (never fully stops -- that would be
    # a disguised terminal/stopping behavior keyed off geometry, not dynamics).
    visibility_min_speed_scale: float = 0.3


@dataclass(frozen=True)
class LimitedCommand:
    """One tick's limited command. `reasons` lists every bound that was active, in
    the order it was applied; empty when the request already satisfied every bound."""

    velocity_ned: np.ndarray
    acceleration_ned: np.ndarray
    yaw_rad: float
    reasons: List[str] = field(default_factory=list)


def _all_finite(*values) -> bool:
    for value in values:
        if value is None:
            continue
        if not np.all(np.isfinite(np.asarray(value, dtype=float))):
            return False
    return True


def visibility_speed_scale(
    center_px: Optional[Sequence[float]],
    image_width_px: Optional[float],
    image_height_px: Optional[float],
    envelope: VehicleEnvelope,
) -> Tuple[float, Optional[str]]:
    """Speed-cap scale in `[visibility_min_speed_scale, 1.0]` from how close
    `center_px` (column, row) is to the image border, and the reason string (or
    `None` once the target is comfortably inside the frame).

    `1.0, None` without pixel geometry (no tracked target this tick): there is
    nothing to protect the observation of.
    """
    if center_px is None or image_width_px is None or image_height_px is None:
        return 1.0, None
    column, row = float(center_px[0]), float(center_px[1])
    half_w, half_h = float(image_width_px) / 2.0, float(image_height_px) / 2.0
    if half_w <= 0.0 or half_h <= 0.0:
        return 1.0, None
    margin_x = min(column, float(image_width_px) - column) / half_w
    margin_y = min(row, float(image_height_px) - row) / half_h
    # 1.0 at the image center, 0.0 (or negative, if already outside) at the border.
    margin = min(margin_x, margin_y)
    threshold = float(envelope.visibility_margin_frac)
    if threshold <= 0.0 or margin >= threshold:
        return 1.0, None
    frac = max(0.0, margin) / threshold
    floor = float(envelope.visibility_min_speed_scale)
    return floor + (1.0 - floor) * frac, "visibility_margin"


def limit_command(
    requested_velocity_ned,
    requested_acceleration_ned,
    requested_yaw_rad: float,
    previous_velocity_ned,
    previous_acceleration_ned,
    previous_yaw_rad: Optional[float],
    dt: Optional[float],
    envelope: VehicleEnvelope,
    center_px: Optional[Sequence[float]] = None,
    image_width_px: Optional[float] = None,
    image_height_px: Optional[float] = None,
) -> LimitedCommand:
    """One coherent velocity/acceleration/yaw command, bounded by `envelope`.

    `previous_*`/`dt` are `None` (or `dt <= 0`) on the first tick: continuity/jerk/
    yaw-rate bounds are skipped (there is no prior command to be continuous with),
    but the envelope/vertical/visibility bounds still apply.
    """
    if not _all_finite(requested_velocity_ned, requested_acceleration_ned, requested_yaw_rad, dt):
        fallback_velocity = (
            np.zeros(3) if previous_velocity_ned is None else np.array(previous_velocity_ned, dtype=float, copy=True)
        )
        fallback_acceleration = (
            np.zeros(3) if previous_acceleration_ned is None
            else np.array(previous_acceleration_ned, dtype=float, copy=True)
        )
        fallback_yaw = 0.0 if previous_yaw_rad is None else float(previous_yaw_rad)
        return LimitedCommand(fallback_velocity, fallback_acceleration, fallback_yaw, ["non_finite_input_rejected"])

    velocity = np.array(requested_velocity_ned, dtype=float, copy=True)
    acceleration = np.array(requested_acceleration_ned, dtype=float, copy=True)
    yaw = float(requested_yaw_rad)
    reasons: List[str] = []

    # Visibility margin first: it scales both the speed cap (below) and the yaw-rate
    # cap (here) -- "speed/turn limiting when the target nears the image edge" -- so
    # a hard turn cannot itself be the thing that loses the target off-frame.
    scale, visibility_reason = visibility_speed_scale(center_px, image_width_px, image_height_px, envelope)

    # Continuity/jerk/yaw-rate bound how far the command may move from the previous
    # (already-hard-capped) one THIS tick; the absolute envelope/vertical/accel-
    # magnitude caps are applied last, as the final safety net, so the returned
    # command always satisfies them even if that means it understeps the continuity
    # bound (for example right after a visibility-margin cap shrinks the speed cap
    # below what continuity alone would have allowed).
    have_continuity = dt is not None and float(dt) > 0.0
    if have_continuity and previous_velocity_ned is not None:
        previous_velocity = np.array(previous_velocity_ned, dtype=float)
        step = velocity - previous_velocity
        step_mag = float(np.linalg.norm(step))
        max_step = float(envelope.max_accel_m_s2) * float(dt)
        if step_mag > max_step:
            velocity = previous_velocity + (step * (max_step / step_mag) if step_mag > 0.0 else step)
            reasons.append("accel_continuity")

    if have_continuity and previous_acceleration_ned is not None:
        previous_acceleration = np.array(previous_acceleration_ned, dtype=float)
        jerk_step = acceleration - previous_acceleration
        jerk_mag = float(np.linalg.norm(jerk_step))
        max_jerk_step = float(envelope.max_jerk_m_s3) * float(dt)
        if jerk_mag > max_jerk_step:
            acceleration = previous_acceleration + (
                jerk_step * (max_jerk_step / jerk_mag) if jerk_mag > 0.0 else jerk_step
            )
            reasons.append("jerk_envelope")

    if have_continuity and previous_yaw_rad is not None:
        delta = float(np.arctan2(np.sin(yaw - previous_yaw_rad), np.cos(yaw - previous_yaw_rad)))
        max_delta = float(envelope.max_yaw_rate_rad_s) * scale * float(dt)
        if abs(delta) > max_delta:
            yaw = float(previous_yaw_rad) + float(np.sign(delta)) * max_delta
            if visibility_reason is not None:
                reasons.append(visibility_reason)
            reasons.append("yaw_rate_envelope")

    if visibility_reason is not None and visibility_reason not in reasons:
        reasons.append(visibility_reason)

    max_speed = float(envelope.max_speed_m_s) * scale
    speed = float(np.linalg.norm(velocity))
    if speed > max_speed:
        velocity = velocity * (max_speed / speed) if speed > 0.0 else velocity
        reasons.append("speed_envelope")

    max_vertical = float(envelope.max_vertical_speed_m_s)
    if abs(velocity[2]) > max_vertical:
        velocity[2] = float(np.clip(velocity[2], -max_vertical, max_vertical))
        reasons.append("vertical_rate_envelope")

    max_accel = float(envelope.max_accel_m_s2)
    accel_mag = float(np.linalg.norm(acceleration))
    if accel_mag > max_accel:
        acceleration = acceleration * (max_accel / accel_mag) if accel_mag > 0.0 else acceleration
        reasons.append("accel_envelope")

    return LimitedCommand(velocity, acceleration, yaw, reasons)
