# FRD-NED alongside FLU-ENU

## Intent

RotorPy integrates and controls in one frame pair, FLU-ENU. A world can instead be created in FRD-NED, the pair PX4 and ArduPilot already speak. Gravity then points the other way, and the controller hovers in that pair. The same physical flight, rewritten back into FLU-ENU, ends where the current flight ends.

## Goals

- The frame pair is chosen when a `World` is constructed. Unset means FLU-ENU, which is today's behavior.
- In FRD-NED, weight is `[0, 0, +mass*g]` with `g = 9.81`. Rotors push along body −z. A level hover is the identity quaternion `[0, 0, 0, 1]` (scalar-last), collective thrust stays positive and about equal to the weight, height stays constant, and the world-frame gravity acceleration is `+z`.
- Position, velocity, and attitude that the integrator and the controller exchange are in the world's pair.
- The examples below can be run in FRD-NED. Compared with the same scenario in FLU-ENU, the end state matches after the conversion in this document.
- `Ardupilot` and the PX4 multirotor still send the autopilot north-east-down and forward-right-down. A FLU-ENU world keeps today's axis rewrite. An FRD-NED world does not apply that rewrite again. Tests cover the outgoing fields with no SITL process.

## Non-goals

- Gymnasium `QuadrotorEnv`, the PPO scripts, RAPTOR, visual servoing, and camera homing.
- Changing `rotorpy/guidance/frames.py`. That module maps NED by negating z only and treats the simulator x axis as north. It is a different map from the one in this document, and homing keeps it.
- Editing vehicle parameter files. They stay authored in FLU.
- Making the 3D plot or the animation draw z downward. The equivalence runs turn plotting and animation off.
- Running the live PX4 or ArduPilot example scripts. Those scripts stay out of the end-state check.

## The two pairs

FLU-ENU, the default:

- World: x east, y north, z up.
- Body: x forward, y left, z up.
- Weight `[0, 0, -mass*g]`. Thrust axis in the body `[0, 0, +1]`.
- Identity attitude: nose east, left wing north.

FRD-NED:

- World: x north, y east, z down.
- Body: x forward, y right, z down.
- Weight `[0, 0, +mass*g]`. Thrust axis in the body `[0, 0, -1]`.
- Identity attitude: nose north, right wing east, belly toward +z.

`g` in the dynamics, the controller, the IMU default, and the wind filters stays `9.81`. PX4's milli-g packing stays on `9.80665`, as it is today.

## Conversion between the pairs

The same physical vector converts as follows. Each map applied twice returns the original.

- World: `(east, north, up)` ↔ `(north, east, -up)`. Matrix `M_enu_to_ned = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]`.
- Body: `(forward, left, up)` ↔ `(forward, -left, -up)`. Matrix `M_flu_to_frd = diag(1, -1, -1)`.

Attitude, scalar-last body-to-world in both pairs:

`R_ned_from_frd = M_enu_to_ned @ R_enu_from_flu @ M_flu_to_frd`

Yaw about the world's up axis, in radians: `yaw_ned = π/2 - yaw_enu`, and `yaw_dot_ned = -yaw_dot_enu`. Yaw `0` in ENU points the nose east. Yaw `0` in NED points the nose north. Those are different headings. A matching flight converts the yaw command. It does not leave the number `0` in both runs and expect the same nose direction.

Body angular rate converts with `M_flu_to_frd`.

These maps live in a new module, `rotorpy/frames.py`, next to a `FrameConvention` enum with values `flu_enu` and `frd_ned`. `rotorpy/guidance/frames.py` is not that module and is not modified.

## Where the choice lives

`World.__init__`, `World.empty`, `World.from_file`, and `World.grid_forest` take `frame: FrameConvention = FrameConvention.FLU_ENU`. The value is stored on `world.frame`. Extents, blocks, and feature points are numbers in that frame. Nothing rewrites them on load. A JSON file may contain a `"frame"` string, `"flu_enu"` or `"frd_ned"`. The `frame` argument of `from_file` overrides the file. Absent both, the world is FLU-ENU.

Vehicle parameter dictionaries stay in FLU. `Multirotor`, `BatchedMultirotor`, `SE3Control`, and `BatchedSE3Control` keep that authored geometry and expose `set_frame(frame)`. The active rotor positions are the authored positions with `M_flu_to_frd` applied when the frame is FRD-NED, and the authored positions when it is FLU-ENU. The allocation matrix is rebuilt from those active positions and the active thrust axis. Inertia follows `I_active = M @ I_authored @ M` with `M = I` or `M = M_flu_to_frd`. Drag coefficients are unchanged. `set_frame` is idempotent: a second call with FRD-NED must not flip the rotors again.

`Environment`, `simulate`, and `simulate_batch` apply `world.frame` by calling `set_frame` on the vehicle, the controller, the IMU, and, when the object has `set_frame`, the estimator. Objects are constructed before the world is attached, which is how the examples are written, so the world's frame wins over whatever frame the object had at construction.

`examples/multimav_multiprocessing_demo.py` builds a `World` and then steps `Multirotor` and `SE3Control` inside a worker that never sees that world. `run_sim` takes the world's frame and calls `set_frame` on the vehicle and controller it creates. With the default frame the worker's numbers stay what they are today.

## Dynamics and the controller

`statedot` uses the active weight and the active thrust axis. The SE3 force is:

- FLU-ENU: add `[0, 0, +g]` per unit mass, and take the thrust direction as `R @ [0, 0, +1]`. Desired body +z is that thrust direction.
- FRD-NED: add `[0, 0, -g]` per unit mass, and take the thrust direction as `R @ [0, 0, -1]`. Desired body +z is the opposite of the desired force, so a hover force along world −z (up) produces the identity attitude.

The same pair of signs is used in `SE3Control`, `BatchedSE3Control`, and the geometric controller inside `Multirotor` / `BatchedMultirotor` (the path used by `cmd_ctatt`, which `examples/batched_simulation.py` selects). Collective thrust for a hover is the positive scalar `mass * g`.

Ground contact follows the same up direction. FLU-ENU: on the ground when `z <= 0.001`, and a downward net force (`Fz < 0`) is cancelled by a +z normal. FRD-NED: on the ground when `z >= -0.001`, and a further-down net force (`Fz > 0`) is cancelled by a −z normal. The listed examples leave ground contact off. The signs still belong to the vehicle so a later caller does not get ENU ground logic in an NED world.

## Sensors, wind, and the camera

The IMU formula is unchanged: specific force is `R_body_from_world @ (a_world - gravity_vector)`. `gravity_vector` is `[0, 0, -9.81]` in FLU-ENU and `[0, 0, +9.81]` in FRD-NED, including `BatchedImu`. At rest in FRD-NED the specific force is along body −z.

`WindUKF` and `WindEKF` take `set_frame`. Their gravity term uses the world's up axis, so a hover with no wind still estimates no wind, and the wind-filter example matches after conversion. Wind-profile classes do not convert. A profile's vector is already in the world frame the caller is using. The equivalence harness converts wind samples. It does not ask `SinusoidWind` to learn a frame.

The pinhole model is unchanged. Camera extrinsics are in the body frame the caller used when building the camera. The harness converts `p_BS` and `R_BS` with the body map. Feature points and vehicle pose convert with the world and attitude maps. The projection code does not grow a frame switch.

## Autopilot bridges

The autopilot frame does not change. PX4 and ArduPilot receive NED position and velocity quantities and FRD specific force and rate quantities. Motor commands coming back are speeds and are not frame-converted.

On a FLU-ENU world the outgoing fields match the current functions on the same state:

- PX4 geodetic uses `enu_to_geodetic(east=x, north=y, up=z)`. Velocity in cm/s is north=`y`, east=`x`, down=`-z`. The quaternion is `Ardupilot._quaternion_rotorpy_to_aerospace`. HIL acceleration negates the IMU y and z and then scales by `9.80665` into milli-g. `HIL_STATE_QUATERNION` angular velocity is `state['w']` with no axis change. `HIL_SENSOR` gyro negates y and z.
- ArduPilot passes `state['x']` and `state['v']` through. The quaternion uses the same aerospace helper. Accelerometer and gyro get `M_flu_to_frd`.

On an FRD-NED world those axis changes are not applied a second time:

- PX4 geodetic uses north=`x`, east=`y`, up=`-z`, then the same latitude/longitude/altitude packing. Velocity in cm/s is north=`x`, east=`y`, down=`z`. The quaternion is the inverse of the simulator body-to-world attitude, packed scalar-first, with no `M_enu_to_ned` or `M_flu_to_frd` multiply. Acceleration stays in milli-g, scaled by `9.80665`; only the y/z negation is skipped. Gyro is packed as the IMU measured it, in rad/s, with no y/z negation.
- ArduPilot still passes position and velocity through, because that is the current function and the state is already NED. The quaternion is the inverse, scalar-first, with no axis sandwich. Accelerometer and gyro are packed as measured.

`HIL_STATE_QUATERNION` angular velocity stays `state['w']` in both pairs. Today's FLU-ENU path already sends it that way. This work does not add a conversion there.

Tests call the conversion and packing helpers with a fixed state. They do not open a MAVLink connection.

## Equivalence check

For each script below, a test runs the scenario twice with plotting and animation off: once in the default FLU-ENU world, once after converting the initial state, the flat outputs (`x`, `x_dot`, `x_ddot`, `yaw`, `yaw_dot`), the world extents and blocks, the wind samples, and any camera extrinsics and feature points into FRD-NED. The FRD-NED record is converted back. At the shared final time the position, velocity, scalar-last attitude, and body rate match.

Shared duration is the example's duration or 5 s, whichever is shorter, except `examples/wind_filter_example.py`, which runs at the script's own durations. Batched waypoint sampling keeps `np.random.seed(10)`. `camera_demo.py` also compares the last frame's projected feature pixels, within 2 pixels. `camera_visualization.py` does not integrate a vehicle. Its check builds one world with a fixed feature seed, converts that world, the parked pose, and the extrinsics, and compares projected feature pixels within 2 pixels. The lens-noise grid and the unseeded per-cell worlds are not part of the match.

Tolerances at that final time: position `1e-2` m, velocity `1e-2` m/s, attitude `1e-2` rad, body rate `1e-2` rad/s.

Scripts in the check:

- `examples/basic_usage.py`
- `examples/camera_demo.py`
- `examples/camera_visualization.py`
- `examples/wind_filter_example.py`
- `examples/batched_simulation.py`
- `examples/multimav_multiprocessing_demo.py`

An untouched run, world frame left unset, keeps today's numbers. The scripts gain no new required argument. `from_file` and `World.empty` already used by these scripts stay FLU-ENU until a caller passes `frame`.

### Wind-filter trials

`examples/wind_filter_example.py` is compared trial by trial, all 50 trials, not as one shortened hover. Each trial keeps the script's two flights: the 30 s calibration along the waypoint list, then the drag fit, then the 10 s hover in `DrydenGust` scored by `WindUKF`. Plotting stays off. The 5 s cap does not apply.

`np.random.seed(0)` is set once, as the script does. The mass, drag, rotor-drag, and wind draws for all 50 trials are taken from that sequence a single time. The FRD-NED run reuses those draws. It does not roll the uniforms again. Scalar parameters (`mass`, `c_Dx`, `c_Dy`, `c_Dz`, `k_d`, `k_z`) are copied. Wind mean and the calibration waypoints are converted with `M_enu_to_ned`. Per-axis wind intensities follow the swapped horizontal axes and an unchanged vertical axis. The gust that reaches the vehicle, and the IMU noise and bias-walk samples, are the FLU-ENU samples with `M_enu_to_ned` or `M_flu_to_frd` applied, so each trial sees the same physical wind and the same physical sensor noise.

The calibration fit removes commanded thrust along the active thrust axis before the absolute value. In FLU-ENU that is `accel_z -= thrust/mass`. In FRD-NED the thrust axis is body −z, so `accel_z += thrust/mass`. Fitted `c_Dx`, `c_Dy`, and `c_Dz` are invariant under that axis flip.

After the FRD-NED trial is mapped back to FLU-ENU, each trial matches on:

- fitted `c_Dx`, `c_Dy`, `c_Dz`, within `1e-3`
- wind RMSE on forward, left, and up, within `1e-2` m/s (the squared error is unchanged by flipping left/right and up/down)
- the last UKF body-wind components, state indexes 6, 7, and 8, within `1e-2` m/s after `M_flu_to_frd`
- the last vehicle position, velocity, attitude, and body rate, at the tolerances above

## Calls made here

- Public type: `FrameConvention` in `rotorpy/frames.py`, string values `flu_enu` and `frd_ned`. Propagation is `set_frame`, called from `Environment`, `simulate`, and `simulate_batch`, not a new argument on every example constructor.
- Parameter files stay FLU. `set_frame` derives the active geometry from a stored FLU copy.
- Wind-profile objects are not frame-aware. The harness converts their samples.
- The pinhole camera is not frame-aware. The harness converts extrinsics and points.
- Equivalence duration is at most 5 s, except the wind-filter script. That check is all 50 trials at 30 s plus 10 s, trial by trial, with thrust removed along the active thrust axis. Tolerances are the four numbers above, plus fitted drag within `1e-3` and wind RMSE and final body-wind within `1e-2` m/s. Camera pixels within 2.
- PX4 on an FRD-NED world still writes `HIL_STATE_QUATERNION` acceleration in milli-g. Only the axis negation is skipped.
- `camera_visualization.py` is a parked render, not a flight. The check is one seeded world, not the noise grid.
- Plotting and animation are out of scope.
- `guidance/frames.py` is left alone.
