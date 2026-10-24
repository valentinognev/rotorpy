"""PX4 ControlMath helpers."""

from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation


def constrain_xy(v0: np.ndarray, v1: np.ndarray, max_norm: float) -> np.ndarray:
    """Return v0 + v1 limited to max_norm, giving v0 priority.

    This is ControlMath::constrainXY. The result is a new array of shape (2,).
    """
    priority = np.array(v0, dtype=float, copy=True)
    secondary = np.array(v1, dtype=float, copy=True)
    total = priority + secondary
    if np.linalg.norm(total) <= max_norm:
        return total

    priority_norm = float(np.linalg.norm(priority))
    if priority_norm >= max_norm:
        return priority * (max_norm / priority_norm)

    if np.linalg.norm(secondary - priority) < 0.001:
        return priority * (max_norm / priority_norm)

    if priority_norm < 0.001:
        secondary_norm = float(np.linalg.norm(secondary))
        return secondary * (max_norm / secondary_norm)

    direction = secondary / np.linalg.norm(secondary)
    along = float(np.dot(direction, priority))
    radius_gap = float(np.dot(priority, priority) - max_norm * max_norm)
    scale = float(np.sqrt(along * along - radius_gap) - along)
    return priority + direction * scale


def limit_tilt(body_unit: np.ndarray, world_unit: np.ndarray, max_angle: float) -> np.ndarray:
    """Return body_unit limited to max_angle radians from world_unit.

    This is ControlMath::limitTilt. The result is a new array of shape (3,).
    body_unit and world_unit are not modified.
    """
    body = np.array(body_unit, dtype=float, copy=True)
    world = np.array(world_unit, dtype=float, copy=True)
    dot = float(np.dot(body, world))
    angle = min(float(np.arccos(np.clip(dot, -1.0, 1.0))), float(max_angle))
    rejection = body - dot * world
    if float(np.dot(rejection, rejection)) < 1.19209290e-7:
        rejection = np.array([1.0, 0.0, 0.0])
    rejection_unit = rejection / np.linalg.norm(rejection)
    return np.cos(angle) * world + np.sin(angle) * rejection_unit


def thrust_from_acceleration(acc_ned: np.ndarray, hover_thrust: float, tilt_max_rad: float, thr_min: float, g: float = 9.81) -> np.ndarray:
    """Return the thrust setpoint for a desired NED acceleration.

    This is PositionControl::_accelerationControl before the later saturation
    step. The result is a new array of shape (3,). acc_ned is not modified.
    """
    acc = np.array(acc_ned, dtype=float, copy=True)
    body_z = np.array([-acc[0], -acc[1], g], dtype=float)
    body_z = body_z / np.linalg.norm(body_z)
    body_z = limit_tilt(body_z, np.array([0.0, 0.0, 1.0]), tilt_max_rad)
    collective = acc[2] * (hover_thrust / g) - hover_thrust
    collective = collective / body_z[2]
    collective = min(collective, -float(thr_min))
    return body_z * collective


def saturate_thrust(thr_sp: np.ndarray, thr_max: float, thr_xy_margin: float) -> np.ndarray:
    """Return thrust limited to thr_max while reserving a horizontal margin.

    This is the saturation half of PositionControl::_velocityControl. It does
    not apply MPC_THR_MIN. The result is a new array of shape (3,). thr_sp is
    not modified.
    """
    thrust = np.array(thr_sp, dtype=float, copy=True)
    thr_xy = thrust[0:2]
    thr_xy_norm = float(np.linalg.norm(thr_xy))
    allocated = min(thr_xy_norm, float(thr_xy_margin))
    vertical_room = float(thr_max) * float(thr_max) - allocated * allocated
    thrust[2] = max(float(thrust[2]), -float(np.sqrt(max(vertical_room, 0.0))))
    horizontal_room = float(thr_max) * float(thr_max) - float(thrust[2]) * float(thrust[2])
    if horizontal_room <= 0.0:
        thrust[0] = 0.0
        thrust[1] = 0.0
    elif thr_xy_norm > 0.0:
        max_xy = float(np.sqrt(horizontal_room))
        if thr_xy_norm > max_xy:
            thrust[0:2] = thr_xy * (max_xy / thr_xy_norm)
    return thrust


def velocity_integral(vel_error: np.ndarray, acc_sp: np.ndarray, thr_sp_before: np.ndarray, thr_sp_after: np.ndarray, vel_int: np.ndarray, i_gain: np.ndarray, p_xy: float, hover_thrust: float, thr_min: float, thr_max: float, dt: float, g: float = 9.81) -> np.ndarray:
    """Return the updated velocity integral with anti-windup.

    This is the anti-windup and integration half of PositionControl::_velocityControl.
    Vertical anti-windup reads thr_sp_before. Horizontal anti-reset reads thr_sp_after.
    The result is a new array of shape (3,). Inputs are not modified.
    """
    error = np.array(vel_error, dtype=float, copy=True)
    integral = np.array(vel_int, dtype=float, copy=True)
    thr_before_z = float(thr_sp_before[2])
    if (thr_before_z >= -float(thr_min) and float(error[2]) >= 0.0) or (
        thr_before_z <= -float(thr_max) and float(error[2]) <= 0.0
    ):
        error[2] = 0.0

    acc_xy_limited = np.array(thr_sp_after[0:2], dtype=float) * (float(g) / float(hover_thrust))
    arw_gain = 2.0 / float(p_xy)
    error[0:2] = error[0:2] - arw_gain * (np.array(acc_sp[0:2], dtype=float) - acc_xy_limited)
    integral = integral + error * np.array(i_gain, dtype=float) * float(dt)

    z = float(integral[2])
    if z > 0.0:
        integral[2] = min(z, float(g))
    elif z < 0.0:
        integral[2] = -min(-z, float(g))
    return integral


def thrust_to_attitude(thr_sp: np.ndarray, yaw: float) -> Tuple[np.ndarray, float]:
    """Return the scalar-last attitude quaternion and body-z thrust.

    This is ControlMath::thrustToAttitude then bodyzToAttitude. Body +z follows
    -thr_sp. The float is thrust_body_z = -norm(thr_sp). thr_sp is not modified.
    """
    thrust = np.array(thr_sp, dtype=float, copy=True)
    body_z = -thrust
    if float(np.dot(body_z, body_z)) < 1.19209290e-7:
        body_z = np.array([0.0, 0.0, 1.0])
    else:
        body_z = body_z / np.linalg.norm(body_z)

    y_c = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    body_x = np.cross(y_c, body_z)
    if body_z[2] < 0.0:
        body_x = -body_x
    if abs(float(body_z[2])) < 1e-6:
        body_x = np.array([0.0, 0.0, 1.0])
    else:
        body_x = body_x / np.linalg.norm(body_x)
    body_y = np.cross(body_z, body_x)
    rotation = np.column_stack((body_x, body_y, body_z))
    quaternion = Rotation.from_matrix(rotation).as_quat()
    return quaternion, -float(np.linalg.norm(thrust))


_CANONICAL_EPS = 1.19209290e-7


def _xyzw_to_wxyz(q_xyzw: np.ndarray) -> np.ndarray:
    quat = np.array(q_xyzw, dtype=float, copy=True)
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=float)


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = (float(left[0]), float(left[1]), float(left[2]), float(left[3]))
    w2, x2, y2, z2 = (float(right[0]), float(right[1]), float(right[2]), float(right[3]))
    return np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dtype=float)


def _quat_inverse(q: np.ndarray) -> np.ndarray:
    norm_sq = float(np.dot(q, q))
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float) / norm_sq


def _component_sign(value: float) -> float:
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _canonical(q: np.ndarray) -> np.ndarray:
    quat = np.array(q, dtype=float, copy=True)
    for component in quat:
        if abs(float(component)) > _CANONICAL_EPS:
            return quat * _component_sign(float(component))
    return quat


def _dcm_z(q: np.ndarray) -> np.ndarray:
    a, b, c, d = (float(q[0]), float(q[1]), float(q[2]), float(q[3]))
    return np.array([
        2.0 * (a * c + b * d),
        2.0 * (c * d - a * b),
        a * a - b * b - c * c + d * d,
    ], dtype=float)


def _quat_from_two_vectors(src: np.ndarray, dst: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Shortest rotation from src to dst, matching PX4 Quaternion(src, dst)."""
    source = np.array(src, dtype=float, copy=True)
    destination = np.array(dst, dtype=float, copy=True)
    cross = np.cross(source, destination)
    dot = float(np.dot(source, destination))
    if float(np.linalg.norm(cross)) < eps and dot < 0.0:
        cross = np.abs(source)
        if cross[0] < cross[1]:
            if cross[0] < cross[2]:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = np.array([0.0, 0.0, 1.0])
        else:
            if cross[1] < cross[2]:
                axis = np.array([0.0, 1.0, 0.0])
            else:
                axis = np.array([0.0, 0.0, 1.0])
        w = 0.0
        cross = np.cross(source, axis)
    else:
        w = dot + float(np.sqrt(float(np.dot(source, source)) * float(np.dot(destination, destination))))
    quat = np.array([w, cross[0], cross[1], cross[2]], dtype=float)
    return quat / np.linalg.norm(quat)


def attitude_rate_setpoint(q_xyzw: np.ndarray, qd_xyzw: np.ndarray, yaw_rate: float, roll_p: float, pitch_p: float, yaw_p: float, yaw_weight: float, rate_limit_rad: np.ndarray) -> np.ndarray:
    """Return the body rate setpoint for a desired attitude.

    This is AttitudeControl::update, including setProportionalGain's division
    of the yaw P gain by yaw_weight when yaw_weight > 1e-4. Quaternions are
    scalar-last [x, y, z, w]. The result is a new array of shape (3,) in FRD
    rad/s. Inputs are not modified.
    """
    q = _xyzw_to_wxyz(q_xyzw)
    qd = _xyzw_to_wxyz(qd_xyzw)
    qd_red = _quat_from_two_vectors(_dcm_z(q), _dcm_z(qd))
    if abs(float(qd_red[1])) > (1.0 - 1e-5) or abs(float(qd_red[2])) > (1.0 - 1e-5):
        qd_red = np.array(qd, dtype=float, copy=True)
    else:
        qd_red = _quat_multiply(qd_red, q)

    q_mix = _canonical(_quat_multiply(_quat_inverse(qd_red), qd))
    mix_w = float(np.clip(q_mix[0], -1.0, 1.0))
    mix_z = float(np.clip(q_mix[3], -1.0, 1.0))
    weight = float(yaw_weight)
    yaw_mix = np.array([
        np.cos(weight * np.arccos(mix_w)),
        0.0,
        0.0,
        np.sin(weight * np.arcsin(mix_z)),
    ], dtype=float)
    qd_mixed = _quat_multiply(qd_red, yaw_mix)

    q_error = _canonical(_quat_multiply(_quat_inverse(q), qd_mixed))
    error = 2.0 * q_error[1:4]
    yaw_gain = float(yaw_p) / weight if weight > 1e-4 else float(yaw_p)
    rate = error * np.array([float(roll_p), float(pitch_p), yaw_gain], dtype=float)
    rate = rate + _dcm_z(_quat_inverse(q)) * float(yaw_rate)
    limits = np.array(rate_limit_rad, dtype=float, copy=True)
    return np.clip(rate, -limits, limits)


def rate_torque(rate: np.ndarray, rate_sp: np.ndarray, angular_accel: np.ndarray, rate_int: np.ndarray, p_gain: np.ndarray, i_gain: np.ndarray, d_gain: np.ndarray, ff_gain: np.ndarray, sat_positive: np.ndarray, sat_negative: np.ndarray, int_limit: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return body torque and the updated rate integral.

    This is RateControl::update. Torque uses the rate error before saturation
    flags. The integral sees that error after a positive flag zeros a positive
    error and a negative flag zeros a negative error, and it always runs.
    P, I, and D are already scaled by the K gains. Both results are new arrays
    of shape (3,). Inputs are not modified.
    """
    measured = np.array(rate, dtype=float, copy=True)
    setpoint = np.array(rate_sp, dtype=float, copy=True)
    accel = np.array(angular_accel, dtype=float, copy=True)
    integral = np.array(rate_int, dtype=float, copy=True)
    proportional = np.array(p_gain, dtype=float, copy=True)
    integral_gain = np.array(i_gain, dtype=float, copy=True)
    derivative = np.array(d_gain, dtype=float, copy=True)
    feedforward = np.array(ff_gain, dtype=float, copy=True)
    limit = np.array(int_limit, dtype=float, copy=True)

    error = setpoint - measured
    torque = proportional * error + integral - derivative * accel + feedforward * setpoint

    integrated_error = np.array(error, dtype=float, copy=True)
    positive = np.array(sat_positive, dtype=bool, copy=True)
    negative = np.array(sat_negative, dtype=bool, copy=True)
    integrated_error = np.where(positive & (integrated_error > 0.0), 0.0, integrated_error)
    integrated_error = np.where(negative & (integrated_error < 0.0), 0.0, integrated_error)

    i_factor = np.maximum(0.0, 1.0 - (integrated_error / np.radians(400.0)) ** 2)
    updated = integral + i_factor * integral_gain * integrated_error * float(dt)
    updated = np.clip(updated, -limit, limit)
    return np.array(torque, dtype=float, copy=True), np.array(updated, dtype=float, copy=True)
