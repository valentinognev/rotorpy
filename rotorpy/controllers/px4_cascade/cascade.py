"""PX4 multicopter controller: rotor allocation and the position, velocity, and attitude cascade.

Default frame is FLU-ENU. ``set_frame`` rebuilds from the authored FLU rotor
positions, so a second call with the same frame does not flip the geometry.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.controllers.px4_cascade.gains import GAINS
from rotorpy.controllers.px4_cascade.math import (
    attitude_rate_setpoint,
    constrain_xy,
    rate_torque,
    saturate_thrust,
    thrust_from_acceleration,
    thrust_to_attitude,
    velocity_integral,
)
from rotorpy.frames import (
    FrameConvention,
    M_FLU_TO_FRD,
    attitude_enu_flu_to_ned_frd,
    attitude_ned_frd_to_enu_flu,
    enu_to_ned,
    flu_to_frd,
    frd_to_flu,
    ned_to_enu,
    yaw_enu_to_ned,
    yaw_rate_enu_to_ned,
)


class Px4Cascade(object):
    """Rotor geometry and the PX4 velocity cascade, matching ``SE3Control`` allocation."""

    def __init__(self, quad_params: dict) -> None:
        self.num_rotors = quad_params["num_rotors"]
        self.rotor_pos = quad_params["rotor_pos"]
        self._rotor_pos_flu = {
            key: np.array(pos, dtype=float, copy=True)
            for key, pos in self.rotor_pos.items()
        }
        self.rotor_dir = quad_params["rotor_directions"]
        self.k_eta = quad_params["k_eta"]
        self.k_m = quad_params["k_m"]
        self.mass = float(quad_params["mass"])
        self.rotor_speed_max = float(quad_params["rotor_speed_max"])
        self.frame = FrameConvention.FLU_ENU
        self.vel_int = np.zeros(3)
        self.rate_int = np.zeros(3)
        self._sat_positive = np.zeros(3, dtype=bool)
        self._sat_negative = np.zeros(3, dtype=bool)
        self._t_previous = None
        self._velocity_ned_previous = None
        self._acceleration_measured = np.zeros(3)
        self._rate_previous = None
        # Body-frame thrust direction: +z in FLU-ENU, -z in FRD-NED.
        self.thrust_axis = np.array([0.0, 0.0, 1.0])
        self._rebuild_allocation()

    def set_frame(self, frame: FrameConvention) -> None:
        """Derive active geometry from the authored FLU copy.

        Calling this twice with the same frame does not flip the rotors again.
        """
        if frame == FrameConvention.FRD_NED:
            body_from_flu = M_FLU_TO_FRD
        elif frame == FrameConvention.FLU_ENU:
            body_from_flu = np.eye(3)
        else:
            raise ValueError("unknown frame convention: %r" % (frame,))

        self.frame = frame
        self.thrust_axis = body_from_flu @ np.array([0.0, 0.0, 1.0])
        self.rotor_pos = {
            key: body_from_flu @ position
            for key, position in self._rotor_pos_flu.items()
        }
        self._rebuild_allocation()

    def _rebuild_allocation(self) -> None:
        """Rebuild f_to_TM from the active rotor positions and thrust axis."""
        k = self.k_m / self.k_eta

        # Thrust and yaw moment both lie on thrust_axis, matching Multirotor.
        self.f_to_TM = np.vstack((
            np.ones((1, self.num_rotors)),
            np.hstack([
                np.cross(self.rotor_pos[key], self.thrust_axis).reshape(-1, 1)[0:2]
                for key in self.rotor_pos
            ]),
            (k * self.thrust_axis[2] * self.rotor_dir).reshape(1, -1),
        ))
        self.TM_to_f = np.linalg.inv(self.f_to_TM)

    def _position_command_from_flat(self, flat: dict) -> dict:
        """Build a position-tier command from a tierless flat output.

        Jerk (``x_dddot``) and snap are not copied. Missing feedforward keys
        stay absent so the position tier keeps its own defaults.
        """
        command = {"tier": "position"}
        if self.frame == FrameConvention.FLU_ENU:
            command["position_ned"] = enu_to_ned(flat["x"])
            if "x_dot" in flat:
                command["velocity_ned"] = enu_to_ned(flat["x_dot"])
            if "x_ddot" in flat:
                command["acceleration_ned"] = enu_to_ned(flat["x_ddot"])
            if "yaw" in flat:
                command["yaw"] = yaw_enu_to_ned(flat["yaw"])
            if "yaw_dot" in flat:
                command["yaw_rate"] = yaw_rate_enu_to_ned(flat["yaw_dot"])
        else:
            command["position_ned"] = np.array(flat["x"], dtype=float, copy=True)
            if "x_dot" in flat:
                command["velocity_ned"] = np.array(flat["x_dot"], dtype=float, copy=True)
            if "x_ddot" in flat:
                command["acceleration_ned"] = np.array(flat["x_ddot"], dtype=float, copy=True)
            if "yaw" in flat:
                command["yaw"] = float(flat["yaw"])
            if "yaw_dot" in flat:
                command["yaw_rate"] = float(flat["yaw_dot"])
        return command

    def update(self, t: float, state: dict, command: dict) -> dict:
        """Run one cascade tier and return motor commands.

        ``tier`` ``"position"`` requires ``position_ned``. Its velocity setpoint
        is the position P term plus optional ``velocity_ned`` (default zeros):
        horizontal ``constrain_xy`` gives the position term priority inside
        ``MPC_XY_VEL_MAX``, and vertical is clamped to
        ``[-MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DN]``. The velocity tier uses
        ``velocity_ned`` with none of those limits. ``acceleration_ned``
        defaults to zeros, ``yaw`` to the current NED yaw, and ``yaw_rate``
        to 0. After the velocity setpoint is chosen, both tiers share the
        velocity, attitude, rate, and motor pipeline.

        ``tier`` ``"attitude"`` requires ``attitude`` (scalar-last NED/FRD
        quaternion, used as given) and ``thrust`` (PX4 normalized collective,
        hover ``0.5``). Optional ``yaw_rate`` defaults to 0 and is passed into
        ``attitude_rate_setpoint``. The position and velocity loops are not
        run. ``cmd_v`` and ``cmd_acc`` are zeros, and the collective uses the
        companion thrust.

        ``tier`` ``"rate"`` requires ``rate_frd`` and ``thrust``. The rate
        setpoint is ``rate_frd``. ``thrust_magnitude`` is the companion
        ``thrust``. The position, velocity, and attitude loops are not run.
        ``cmd_v`` and ``cmd_acc`` are zeros, and the collective uses the
        companion thrust.

        The first call integrates for 0.01 s. Later calls use ``t - t_previous``.
        Measured acceleration is a first-order low-pass of the NED velocity
        derivative and is zero on the first sample. Rate D is
        ``(rate_frd - previous_rate_frd) / dt`` and is zero on the first sample.
        Saturation flags for the next rate step are the signs of commanded
        moment minus achieved moment after the thrust clamp. Achieved moment
        is ``(f_to_TM @ clamped_thrusts)[1:4]`` in the active body frame. In
        FLU-ENU that residual is mapped with ``flu_to_frd`` before the sign.
        A residual at or below 1e-6 N·m is unsaturated. Flags start cleared.

        A command with ``x`` and no ``tier`` runs the position tier. In
        FLU-ENU, ``x``, ``x_dot``, and ``x_ddot`` go through ``enu_to_ned``,
        ``yaw`` through ``yaw_enu_to_ned``, and ``yaw_dot`` through
        ``yaw_rate_enu_to_ned``. In FRD-NED those values are already PX4 NED
        and are not mapped. Jerk and snap are ignored.
        """
        tier = command.get("tier")
        if tier is None and "x" in command:
            command = self._position_command_from_flat(command)
            tier = command["tier"]
        if tier == "position":
            if "position_ned" not in command:
                raise ValueError("position tier requires 'position_ned'")
        elif tier == "attitude":
            if "attitude" not in command or "thrust" not in command:
                raise ValueError("attitude tier requires 'attitude' and 'thrust'")
        elif tier == "rate":
            if "rate_frd" not in command or "thrust" not in command:
                raise ValueError("rate tier requires 'rate_frd' and 'thrust'")
        elif tier != "velocity":
            raise ValueError("unsupported command tier: %r" % (tier,))

        if self._t_previous is None:
            dt = 0.01
        else:
            dt = float(t) - self._t_previous
        self._t_previous = float(t)
        if self.frame == FrameConvention.FRD_NED:
            position = np.array(state["x"], dtype=float, copy=True)
            velocity = np.array(state["v"], dtype=float, copy=True)
            q_ned = np.array(state["q"], dtype=float, copy=True)
            rate = np.array(state["w"], dtype=float, copy=True)
        else:
            position = enu_to_ned(state["x"])
            velocity = enu_to_ned(state["v"])
            q_ned = attitude_enu_flu_to_ned_frd(state["q"])
            rate = flu_to_frd(state["w"])

        if tier == "rate":
            # rate_frd is the setpoint. No position, velocity, or attitude loop.
            rate_sp = np.array(command["rate_frd"], dtype=float, copy=True)
            thrust_magnitude = float(command["thrust"])
            vel_sp = np.zeros(3)
            acc = np.zeros(3)
            q_sp = np.array(q_ned, dtype=float, copy=True)
        elif tier == "attitude":
            # Companion attitude is already scalar-last NED/FRD. Do not convert it.
            q_sp = np.array(command["attitude"], dtype=float, copy=True)
            if "yaw_rate" in command:
                yaw_rate = float(command["yaw_rate"])
            else:
                yaw_rate = 0.0
            vel_sp = np.zeros(3)
            acc = np.zeros(3)
            thrust_magnitude = float(command["thrust"])
        else:
            if tier == "position":
                position_ned = np.array(command["position_ned"], dtype=float, copy=True)
                if "velocity_ned" in command:
                    velocity_ff = np.array(command["velocity_ned"], dtype=float, copy=True)
                else:
                    velocity_ff = np.zeros(3)
                vel_from_position = (position_ned - position) * np.array([
                    GAINS["MPC_XY_P"],
                    GAINS["MPC_XY_P"],
                    GAINS["MPC_Z_P"],
                ], dtype=float)
                vel_xy = constrain_xy(
                    vel_from_position[0:2],
                    velocity_ff[0:2],
                    GAINS["MPC_XY_VEL_MAX"],
                )
                vel_z = float(np.clip(
                    vel_from_position[2] + velocity_ff[2],
                    -GAINS["MPC_Z_VEL_MAX_UP"],
                    GAINS["MPC_Z_VEL_MAX_DN"],
                ))
                vel_sp = np.array([vel_xy[0], vel_xy[1], vel_z], dtype=float)
            else:
                vel_sp = np.array(command["velocity_ned"], dtype=float, copy=True)
            if "acceleration_ned" in command:
                acc_ff = np.array(command["acceleration_ned"], dtype=float, copy=True)
            else:
                acc_ff = np.zeros(3)
            if "yaw" in command:
                yaw = float(command["yaw"])
            else:
                yaw = float(Rotation.from_quat(q_ned).as_euler("ZYX")[0])
            if "yaw_rate" in command:
                yaw_rate = float(command["yaw_rate"])
            else:
                yaw_rate = 0.0

            p_vel = np.array([
                GAINS["MPC_XY_VEL_P_ACC"],
                GAINS["MPC_XY_VEL_P_ACC"],
                GAINS["MPC_Z_VEL_P_ACC"],
            ], dtype=float)
            d_vel = np.array([
                GAINS["MPC_XY_VEL_D_ACC"],
                GAINS["MPC_XY_VEL_D_ACC"],
                GAINS["MPC_Z_VEL_D_ACC"],
            ], dtype=float)
            i_vel = np.array([
                GAINS["MPC_XY_VEL_I_ACC"],
                GAINS["MPC_XY_VEL_I_ACC"],
                GAINS["MPC_Z_VEL_I_ACC"],
            ], dtype=float)
            vel_error = vel_sp - velocity
            if self._velocity_ned_previous is None:
                self._velocity_ned_previous = np.array(velocity, dtype=float, copy=True)
            else:
                cutoff = float(GAINS["MPC_VELD_LP"])
                alpha = dt / (dt + 1.0 / (2.0 * np.pi * cutoff))
                velocity_derivative = (velocity - self._velocity_ned_previous) / dt
                self._acceleration_measured = (
                    alpha * velocity_derivative
                    + (1.0 - alpha) * self._acceleration_measured
                )
                self._velocity_ned_previous = np.array(velocity, dtype=float, copy=True)
            acceleration_measured = self._acceleration_measured
            acc = vel_error * p_vel + self.vel_int - acceleration_measured * d_vel + acc_ff

            thr_before = thrust_from_acceleration(
                acc,
                GAINS["MPC_THR_HOVER"],
                np.deg2rad(GAINS["MPC_TILTMAX_AIR"]),
                GAINS["MPC_THR_MIN"],
            )
            thr_after = saturate_thrust(
                thr_before,
                GAINS["MPC_THR_MAX"],
                GAINS["MPC_THR_XY_MARG"],
            )
            self.vel_int = velocity_integral(
                vel_error,
                acc,
                thr_before,
                thr_after,
                self.vel_int,
                i_vel,
                GAINS["MPC_XY_VEL_P_ACC"],
                GAINS["MPC_THR_HOVER"],
                GAINS["MPC_THR_MIN"],
                GAINS["MPC_THR_MAX"],
                dt,
            )

            q_sp, thrust_body_z = thrust_to_attitude(thr_after, yaw)
            thrust_magnitude = -float(thrust_body_z)
        if tier != "rate":
            rate_sp = attitude_rate_setpoint(
                q_ned,
                q_sp,
                yaw_rate,
                GAINS["MC_ROLL_P"],
                GAINS["MC_PITCH_P"],
                GAINS["MC_YAW_P"],
                GAINS["MC_YAW_WEIGHT"],
                np.deg2rad([
                    GAINS["MC_ROLLRATE_MAX"],
                    GAINS["MC_PITCHRATE_MAX"],
                    GAINS["MC_YAWRATE_MAX"],
                ]),
            )
        if self._rate_previous is None:
            angular_accel = np.zeros(3)
        else:
            angular_accel = (rate - self._rate_previous) / dt
        self._rate_previous = np.array(rate, dtype=float, copy=True)
        torque, self.rate_int = rate_torque(
            rate,
            rate_sp,
            angular_accel,
            self.rate_int,
            np.array([
                GAINS["MC_ROLLRATE_P"],
                GAINS["MC_PITCHRATE_P"],
                GAINS["MC_YAWRATE_P"],
            ], dtype=float),
            np.array([
                GAINS["MC_ROLLRATE_I"],
                GAINS["MC_PITCHRATE_I"],
                GAINS["MC_YAWRATE_I"],
            ], dtype=float),
            np.array([
                GAINS["MC_ROLLRATE_D"],
                GAINS["MC_PITCHRATE_D"],
                GAINS["MC_YAWRATE_D"],
            ], dtype=float),
            np.array([
                GAINS["MC_ROLLRATE_FF"],
                GAINS["MC_PITCHRATE_FF"],
                GAINS["MC_YAWRATE_FF"],
            ], dtype=float),
            self._sat_positive,
            self._sat_negative,
            np.array([
                GAINS["MC_RR_INT_LIM"],
                GAINS["MC_PR_INT_LIM"],
                GAINS["MC_YR_INT_LIM"],
            ], dtype=float),
            dt,
        )

        if self.frame == FrameConvention.FLU_ENU:
            moment = frd_to_flu(torque)
            cmd_q = attitude_ned_frd_to_enu_flu(q_sp)
            cmd_w = frd_to_flu(rate_sp)
            cmd_v = ned_to_enu(vel_sp)
            cmd_acc = ned_to_enu(acc)
        else:
            moment = np.array(torque, dtype=float, copy=True)
            cmd_q = np.array(q_sp, dtype=float, copy=True)
            cmd_w = np.array(rate_sp, dtype=float, copy=True)
            cmd_v = np.array(vel_sp, dtype=float, copy=True)
            cmd_acc = np.array(acc, dtype=float, copy=True)

        collective_n = thrust_magnitude / GAINS["MPC_THR_HOVER"] * self.mass * 9.81
        wrench = np.array([
            collective_n,
            moment[0],
            moment[1],
            moment[2],
        ], dtype=float)
        thrusts = self.TM_to_f @ wrench
        max_thrust = self.k_eta * self.rotor_speed_max ** 2
        thrusts = np.clip(thrusts, 0.0, max_thrust)
        speeds = np.sqrt(thrusts / self.k_eta)
        # Next rate step: FRD sign of commanded minus achieved moment.
        achieved_moment = (self.f_to_TM @ thrusts)[1:4]
        residual = moment - achieved_moment
        if self.frame == FrameConvention.FLU_ENU:
            residual = flu_to_frd(residual)
        self._sat_positive = np.greater(residual, 1e-6)
        self._sat_negative = np.less(residual, -1e-6)
        return {
            "cmd_motor_speeds": speeds,
            "cmd_motor_thrusts": thrusts,
            "cmd_thrust": float(collective_n),
            "cmd_moment": moment,
            "cmd_q": cmd_q,
            "cmd_w": cmd_w,
            "cmd_v": cmd_v,
            "cmd_acc": cmd_acc,
        }
