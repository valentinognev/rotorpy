# PX4 cascade for the Hummingbird

## Intent

The homing loop already decides a NED velocity, a NED acceleration, and a yaw. `SE3Control` turns that into motor speeds with a geometric law. The flight controller on the real side of this work is the PX4 multicopter cascade: position, velocity, attitude, and body rate, each able to take the companion command while the loops below it still run. Homing is the first companion, and it commands the velocity tier, with its acceleration added on top of the velocity loop.

## Goals

- One controller, `Px4Cascade`, that outputs `cmd_motor_speeds` for the existing Hummingbird.
- Four companion tiers: position, velocity, attitude, and rate. A tier commands all three axes. Loops above the chosen tier do not add a setpoint.
- Homing flies on the velocity tier: `HomingCommand.velocity_ned`, `HomingCommand.acceleration_ned` as the acceleration feedforward, and `HomingCommand.yaw_rad`. Yaw rate is zero.
- The loop equations and the numeric gains are the ones in [HummingbirdTeam/Firmware](https://github.com/HummingbirdTeam/Firmware) at `795b31a5c7` (the `master` tip). That tree is PX4. It has no Hummingbird airframe and no gain overrides, so the numbers are that tree's multicopter defaults. Generic quad X (`4001_quad_x`) only loads `rc.mc_defaults`.
- An unlocked homing cruise (the existing 20 s, no-lock flight) stays off the ground.

## Non-goals

- EKF, commander, navigator, RC sticks, expo curves, failsafe, land detector, takeoff ramp, battery scaling, hover-thrust estimation, and jerk-limited trajectory generation.
- A separate acceleration-only tier. Acceleration is a feedforward on the position and velocity tiers.
- Replacing `SE3Control` anywhere except the homing runner.
- A batched controller.
- Changing the homing guidance law, the speed cap, or `hummingbird_params.py`.
- Per-axis tier mixing. North and east are always commanded together, and a tier includes z.

## Constraints

- The cascade math is right-handed NED world and FRD body, matching PX4. Quaternions at the RotorPy boundary stay scalar-last `[x, y, z, w]`. PX4's `[w, x, y, z]` never crosses that boundary.
- The simulator state is converted with `rotorpy.frames`. Default frame is FLU-ENU: simulator `(x, y, z)` is east, north, up. `enu_to_ned`, `flu_to_frd`, and `attitude_enu_flu_to_ned_frd` produce the NED/FRD state. `set_frame(FRD_NED)` means the state is already NED/FRD and those maps are skipped. A z-only flip is not used. `(north, east, up)` is left-handed, and the PX4 attitude law needs a rotation.
- Homing commands are in the guidance NED of `rotorpy.guidance.frames`, where simulator `(x, y, z)` is named `(north, east, up)` and only `z` is flipped. That is not the NED above. The runner converts a homing command into PX4 NED before `update`: velocity and acceleration `(n, e, d)` become `(e, n, d)`, and yaw becomes `pi/2 - yaw`. At the no-lock start the converted velocity matches `enu_to_ned` of the initial simulator velocity, and the converted yaw matches the converted initial attitude. Body FRD still flips simulator-body `y` and `z`.
- The homing world stays the unset FLU-ENU world. Motor speeds are scalars. Moments are mixed in the active body frame with the same rotor map `SE3Control` builds.
- `update(t, state, command)` returns the same keys `SE3Control.update` returns. `cmd_motor_speeds` is what `Multirotor` integrates. `cmd_thrust` is Newtons along the active thrust axis (positive when rotors lift the vehicle). `cmd_moment` is Newton-metres in the active body frame.
- Simulation step for homing stays 0.01 s. Integrators and filters use the `dt` between `update` calls. The first call uses `dt` of that same step and treats the previous sample as equal to the current one, so the derivative terms start at zero.
- Gains below are the firmware defaults. Do not retune them to make a test pass.

## What the companion sends

`command["tier"]` is `"position"`, `"velocity"`, `"attitude"`, or `"rate"`.

| Tier | Required keys | Loops that run |
|---|---|---|
| `position` | `position_ned` (3,) | position, velocity, attitude, rate |
| `velocity` | `velocity_ned` (3,) | velocity, attitude, rate |
| `attitude` | `attitude` quaternion, `thrust` | attitude, rate |
| `rate` | `rate_frd` (3,) rad/s, `thrust` | rate |

Optional on `position` and `velocity`: `velocity_ned` (position tier only, feedforward, default zeros), `acceleration_ned` (default zeros), `yaw` (radians, default the current NED yaw), `yaw_rate` (default 0). Optional on `attitude`: `yaw_rate` (default 0), added the same way the attitude loop adds yaw rate.

`thrust` is PX4's normalized collective. Hover is `0.5`. It is the positive magnitude PX4 stores as `-thrust_body[2]`.

A command that has `"x"` and no `"tier"` is a position-tier flat output in the simulator frame: `x`, `x_dot`, `x_ddot`, `yaw`, `yaw_dot`. Jerk and snap are ignored. This is so `Environment` can pass its usual flat output.

Homing builds a `velocity` command in PX4 NED from the homing command, using the horizontal swap and yaw conversion above, and does not go through `flat_output`. `flat_output` stays as it is for the guidance tests.

## Position and velocity

State and setpoints in this section are NED. `g = 9.81`.

Position P, only on the position tier. Otherwise this term is absent:

`vel_from_position = (position_sp - position) * [MPC_XY_P, MPC_XY_P, MPC_Z_P]`

On the position tier the velocity setpoint starts at `velocity_ned` (the feedforward, or zeros). The position term is added on every finite axis. The horizontal part is then limited to `MPC_XY_VEL_MAX` with the PX4 priority in `ControlMath::constrainXY`: the position term keeps priority over the feedforward. Vertical velocity is clamped to `[-MPC_Z_VEL_MAX_UP, MPC_Z_VEL_MAX_DN]`.

On the velocity tier the velocity setpoint is `velocity_ned` unchanged. `MPC_XY_VEL_MAX`, `MPC_Z_VEL_MAX_UP`, and `MPC_Z_VEL_MAX_DN` do not apply. Homing commands a component of 19 m/s, including downward, and those caps are 12 m/s horizontal and 1 m/s down.

Velocity PID:

`acc = (velocity_sp - velocity) * P + vel_int - acceleration_measured * D + acceleration_ned`

`P` is `[MPC_XY_VEL_P_ACC, MPC_XY_VEL_P_ACC, MPC_Z_VEL_P_ACC]`. `D` is the matching `*_VEL_D_ACC` pair. `acceleration_ned` is the feedforward and is absent when the companion did not send it.

`acceleration_measured` is a first-order low-pass of `(velocity - previous_velocity) / dt`, cutoff `MPC_VELD_LP` hertz, state stored on the controller. The first sample is zero.

Acceleration to thrust and tilt, from `PositionControl::_accelerationControl` in that firmware:

- `body_z = normalize([-acc_x, -acc_y, g])`, then `limitTilt` toward world up `[0, 0, 1]` by `MPC_TILTMAX_AIR` degrees. `limitTilt` is `ControlMath::limitTilt`.
- `collective = acc_z * (MPC_THR_HOVER / g) - MPC_THR_HOVER`
- `collective = min(collective / body_z_z, -MPC_THR_MIN)`
- `thr_sp = body_z * collective`

Thrust is then saturated as in `_velocityControl`: keep a horizontal margin of `MPC_THR_XY_MARG`, give the remaining authority to vertical thrust, then give whatever is left to horizontal thrust. Limits are `MPC_THR_MIN` and `MPC_THR_MAX`.

Velocity integrator, updated after that saturation, from the same function:

- Vertical error is zeroed when the vertical thrust is on its limit and the error points further into that limit.
- Horizontal error uses the tracking anti-reset term: `acc_xy_limited = thr_sp_xy * (g / MPC_THR_HOVER)`, `arw_gain = 2 / MPC_XY_VEL_P_ACC`, and the horizontal velocity error loses `arw_gain * (acc_xy - acc_xy_limited)`.
- `vel_int += error * I * dt` with `I = [MPC_XY_VEL_I_ACC, MPC_XY_VEL_I_ACC, MPC_Z_VEL_I_ACC]`.
- `vel_int_z` is clamped to `[-g, g]`.

Yaw setpoint is the companion yaw, or the current yaw when the companion omitted it. Yaw rate feedforward is the companion yaw rate or 0. Attitude comes from `ControlMath::thrustToAttitude(-thr_sp` is not the call; the firmware calls `thrustToAttitude(thr_sp, yaw)` which builds the attitude from `-thr_sp` as the body +z axis in NED, and sets `thrust_body[2] = -norm(thr_sp)`.

## Attitude

From `AttitudeControl::update` in that firmware. `q` is the current NED/FRD attitude. `qd` is the thrust attitude on the position and velocity tiers, and the companion quaternion on the attitude tier.

Reduced attitude `qd_red` rotates the current body z to the desired body z, then is written back into the world frame by multiplying by `q`. The 180° corner case in the firmware (the reduced quaternion's x or y component within `1e-5` of ±1) uses `qd` itself.

Full and reduced attitudes are mixed with `MC_YAW_WEIGHT`: the yaw part of `qd_red.inverse * qd` is scaled by that weight and composed back onto `qd_red`. The proportional gain on yaw passed into this mix is `MC_YAW_P / MC_YAW_WEIGHT` because the firmware divides the yaw gain by the weight once, up front.

Attitude error is `2 * imag(canonical(q.inverse * qd_mixed))`. Body-rate setpoint is that error times `[MC_ROLL_P, MC_PITCH_P, yaw_gain]`. The yaw-rate feedforward is added as `q.inverse` applied to the world z axis, times `yaw_rate`. Each axis is then clamped to `MC_ROLLRATE_MAX`, `MC_PITCHRATE_MAX`, `MC_YAWRATE_MAX` in degrees per second, converted to radians.

## Rate

From `RateControl::update`. The firmware multiplies `MC_*RATE_K` into P, I, and D before this formula. With every `K` at `1`, the gains below are the numbers that enter the formula.

`torque = P * (rate_sp - rate) + rate_int - D * angular_accel + FF * rate_sp`

`rate` and `rate_sp` are FRD rad/s. `angular_accel` is `(rate - previous_rate) / dt` in FRD, and zero on the first sample. `torque` is Newton-metres in FRD. `vehicle_torque_setpoint.xyz` in that firmware is documented as Newton-metres, so the gain numbers are used in those units with no extra scale.

The integrator updates only in flight. This controller is always in flight. Per axis, a positive saturation flag from the previous mix zeros a positive rate error before integrating, and a negative flag zeros a negative error. The i-factor is `max(0, 1 - (rate_error / radians(400))^2)`. The increment is `i_factor * I * rate_error * dt`, then clamped to `±MC_RR_INT_LIM`, `±MC_PR_INT_LIM`, `±MC_YR_INT_LIM`.

## Motors

`collective_N = thrust_magnitude / MPC_THR_HOVER * mass * g`, where `thrust_magnitude` is `-thrust_body[2]` on the position, velocity, and attitude tiers, and the companion `thrust` on the rate tier. At hover (`0.5`) this equals the weight.

That collective is positive along the active thrust axis. The FRD torque is mapped into the active body frame (`frd_to_flu` when the frame is FLU-ENU). Rotor thrusts come from the inverse of the same `f_to_TM` map `SE3Control` builds for this vehicle and frame. Each thrust is clamped to `[0, k_eta * rotor_speed_max^2]`. Speed is `sqrt(thrust / k_eta)`.

The saturation flags for the next rate step are the signs of `commanded_moment - achieved_moment` on each active-body axis after that clamp. An axis with a zero residual is unsaturated. The first step is unsaturated.

## Gains

From `mc_pos_control_params.c`, `mc_att_control_params.c`, and `mc_rate_control_params.c` at `795b31a5c7`.

| Name | Value |
|---|---|
| `MPC_XY_P` | 0.95 |
| `MPC_Z_P` | 1.0 |
| `MPC_XY_VEL_P_ACC` | 1.8 |
| `MPC_XY_VEL_I_ACC` | 0.4 |
| `MPC_XY_VEL_D_ACC` | 0.2 |
| `MPC_Z_VEL_P_ACC` | 4.0 |
| `MPC_Z_VEL_I_ACC` | 2.0 |
| `MPC_Z_VEL_D_ACC` | 0.0 |
| `MPC_XY_VEL_MAX` | 12 m/s |
| `MPC_Z_VEL_MAX_UP` | 3 m/s |
| `MPC_Z_VEL_MAX_DN` | 1 m/s |
| `MPC_TILTMAX_AIR` | 45° |
| `MPC_THR_MIN` | 0.12 |
| `MPC_THR_MAX` | 1.0 |
| `MPC_THR_HOVER` | 0.5 |
| `MPC_THR_XY_MARG` | 0.3 |
| `MPC_VELD_LP` | 5 Hz |
| `MC_ROLL_P`, `MC_PITCH_P` | 6.5 |
| `MC_YAW_P` | 2.8 |
| `MC_YAW_WEIGHT` | 0.4 |
| `MC_ROLLRATE_MAX`, `MC_PITCHRATE_MAX` | 220°/s |
| `MC_YAWRATE_MAX` | 200°/s |
| `MC_ROLLRATE_P`, `MC_PITCHRATE_P` | 0.15 |
| `MC_ROLLRATE_I`, `MC_PITCHRATE_I` | 0.2 |
| `MC_ROLLRATE_D`, `MC_PITCHRATE_D` | 0.003 |
| `MC_YAWRATE_P` | 0.2 |
| `MC_YAWRATE_I` | 0.1 |
| `MC_YAWRATE_D` | 0.0 |
| `MC_ROLLRATE_FF`, `MC_PITCHRATE_FF`, `MC_YAWRATE_FF` | 0 |
| `MC_ROLLRATE_K`, `MC_PITCHRATE_K`, `MC_YAWRATE_K` | 1 |
| `MC_RR_INT_LIM`, `MC_PR_INT_LIM`, `MC_YR_INT_LIM` | 0.30 |

## Homing runner

`examples/camera_homing.jsonc` selects the inner loop. `pid`, the shipped value, constructs `Px4Cascade(quad_params)`. Each step calls `update` with tier `velocity` and the homing velocity, acceleration, and yaw after the PX4 NED conversion above. `geometric` constructs `SE3Control` and passes `flat_output`. The vehicle stays `cmd_motor_speeds`. Guidance, cameras, stop rules, and the recorded history are unchanged. The same file holds the start, the target, both cameras, and the timing.

The no-lock flight commands the initial horizontal 10 m/s cruise and zero vertical velocity for 20 s. It must still time out in the air, which is what `test_never_lock_times_out_at_20_seconds` checks. That test stays marked `homing_e2e` and is not part of the default suite.

## Builder calls

- State conversion uses `rotorpy.frames` FLU-ENU, not the homing z-only flip. Homing's `(n, e, d)` and yaw are converted as specified above so the velocity loop sees one NED. Settled from the two frame modules already in the tree: PX4 needs a right-handed attitude, and the cruise must start with zero horizontal velocity error.
- Equation source is HummingbirdTeam/Firmware `795b31a5c7`, not a later PX4 `main`. The rate `K` gains are folded in as that tree's `MulticopterRateControl` does, and at `K = 1` the formula is unchanged.
- Velocity and descent caps apply on the position tier only. The velocity tier passes the companion velocity through, so homing is not clipped to 12 m/s or 1 m/s down. Tilt and thrust limits apply on every tier.
- `MPC_THR_HOVER = 0.5` stays the firmware number. Newtons are `thrust_magnitude / 0.5 * mass * g`, so hover thrust is the Hummingbird weight even though half of `rotor_speed_max` is not hover on this motor curve.
- Rate-controller torque is Newton-metres, as `vehicle_torque_setpoint` states. No second scale onto the hummingbird inertia.
- Measured acceleration for the velocity D term is the 5 Hz low-pass of velocity finite difference. Rate D uses the finite difference of body rate. Neither reads a sensor message the simulator does not have.
- The integrator runs on every `update`. There is no landed flag.
- Saturation flags come from the previous mix.
- A flat output with no `tier` is the position tier in the simulator frame.
