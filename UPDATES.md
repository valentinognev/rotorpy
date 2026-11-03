# Updates

## 4.6.0 - Timestamp-aware tracker and independent setpoint streaming (plan Phase 3)
- New pure, unit-tested `rotorpy/guidance/` modules, moved out of `Dockerfiles/camera_homing_tracker.py`: `observation_buffer.py` (per-camera depth-1 latest-observation slots, independent freshness so one camera's new frame never refreshes the other's cached one; `decode_image` for rgb8/bgr8/rgba8/bgra8/mono8 with row-stride handling, explicit error on unsupported encodings); `vehicle_state.py` (`ClockModel` distinguishing sim time from wall time and flagging pause/reset/backward-jump; `AttitudeHistory` ring buffer with slerp interpolation and an in-range flag; `VehicleState` aggregating PX4-style `ATTITUDE_QUATERNION`/`LOCAL_POSITION_NED` samples with wall-clock ages); `setpoint_stream.py` (`SetpointStream`: independent periodic publisher, callable-based with an injectable clock, expiry and a documented failure response, stats for publish interval and observation age); `recording_worker.py` (bounded queue + background thread so a slow sink cannot block the control path, drop-oldest-on-full with a counter); `watchdog.py` (wall-time staleness watchdogs for heartbeat/telemetry/image comms links).
- `origin_alignment.EstimatorOriginAlignment` gained `attitude_to_world`/`attitude_to_estimator` (quaternion crosses the yaw-offset boundary the same way positions and vectors already did) and `height_above_ground_world` (height from a position already in the world frame).
- The tracker now reads vehicle attitude/position/velocity from PX4 telemetry via `VehicleState` (using `EstimatorOriginAlignment` wherever estimator-frame data crosses into world-frame guidance/telemetry), not Gazebo's model-state truth; Gazebo truth is read only for recording and outcome scoring (`_pose_truth`), gated so the service is not even called when neither telemetry nor the recorder is active. Setpoint publishing runs on its own `rospy.Timer` cadence, independent of the camera-processing loop; a stale or absent guidance command stops the normal command regardless of heartbeat/armed/offboard state. Recording (telemetry ticks, outcome) is submitted to a `RecordingWorker` off the control path.
- `homing_config.py` gained `PROPOSED_HEARTBEAT_TIMEOUT_S`/`PROPOSED_TELEMETRY_TIMEOUT_S`/`PROPOSED_IMAGE_TIMEOUT_S` and `watchdog_timeouts()`, alongside the per-module proposed defaults (`vehicle_state.PROPOSED_CLOCK_*`, `vehicle_state.PROPOSED_ATTITUDE_HISTORY_SAMPLES`, `setpoint_stream.PROPOSED_SETPOINT_EXPIRY_S`, `recording_worker.PROPOSED_RECORDING_QUEUE_DEPTH`).
- Tests: `test_observation_buffer.py`, `test_vehicle_state.py`, `test_setpoint_stream.py`, `test_recording_worker.py` (real background thread, `threading.Event`-synchronized, no sleeps), `test_watchdog.py`, `test_homing_config_watchdogs.py`; `test_homing_origin_alignment.py` extended for the new attitude/height methods. `test_gazebo_homing_startup.py` updated throughout for the new tracker internals (PX4-estimate-driven `step()`, `_poll_mavlink`, `_publish_setpoint`/`_send`, `_pose_truth`), preserving its existing coverage (gravity-link coverage, motor-ready thresholds, startup-hold timing, heartbeat persistence, telemetry/outcome recording, replay round-trip, vehicle-sourced startup heading).
- Fix round 1 (review findings): extracted the setpoint command/stamp decision into a pure `guidance/step_decision.py::command_and_stamp` and fixed `step()` restamping the setpoint stream's held command every tick even when holding stale data, so a dead camera/PX4 estimate now lets the stream actually expire; wired `any_stale`/`mark_fatal` (comms watchdog staleness and fatal `step()` failures now stop the normal command, sticky); `AttitudeHistory.record` tolerates duplicate/non-increasing sim timestamps (no more interpolation crash); `SetpointStream` gained a lock (`peek`/`stats_snapshot`, atomic command+time), and the tracker gained `_mav_lock`/`_flags_lock` for cross-thread pymavlink/flag safety; `SetpointStats` now counts only real publishes and is recorded in telemetry (`TickRecord.setpoint_stats`); `VehicleState`'s clock-discontinuity flag is sticky until `sample_at` consumes it, and `step()` resets a new `HomingGuidance.reset_derivative_history()` on one; a fresh frame from one camera no longer drops the other camera's still-recent cached frame (`observation_buffer.select_fresh_or_recent`); `setpoint_stream.PROPOSED_FAILURE_COMMAND` is now a named, documented default; `decode_image` no longer demands trailing stride padding on the last row; the image watchdog is touched only after a successful decode.

## 4.5.0 - Gazebo camera geometry and one authoritative scene (plan Phase 2)
- Fixed the off-axis sign error: the Gazebo path now converts pixels camera -> body FRD -> NED (`camera_geometry.CameraMount`, `pixel_ray_ned`, `project_ned_point`, `mount_from_gazebo_pose`), so image-right of a north-facing vehicle is east (the old path was mirrored, ~23 deg off at 224 px). Each camera's translation (forward: 0.36 m ahead, 0.05 m up) is in the ray origin; the sightline is from the body origin to the ray's hit on the target-reference plane (`sightline.reference_relative_position`).
- `HomingGuidance` takes `forward_mount`/`upward_mount` for that geometry. Without mounts it keeps the legacy RotorPy example convention (mirrored body +y, `_sightline_ned`) as an explicit compatibility boundary; `gazebo_bridge.attitude_homing_to_ned_frd` adapts the homing quaternion. `guidance_from_config` is the Gazebo (mounts) path.
- Target position no longer reaches guidance: `initial_velocity_from_config` is gone; before lock the guidance flies `initial_speed_m_s` along the vehicle's own heading at its first update (runner and tracker too; tracker startup yaw = measured heading).
- `scene.TargetReference`: ground contact vs visible centre (cylinder centre is height/2 above contact); config adds `gazebo_target_radius_m`/`gazebo_target_height_m`; guidance tracks the visible centre. `gazebo_scene_from_config` gives the replay page the cylinder's bounding box and the real camera mounts (`forward/upward_camera_position_flu_m`).
- `origin_alignment.EstimatorOriginAlignment` (+ config `estimator_origin_world_ned`, `estimator_yaw_offset_deg`, PROPOSED): world <-> estimator translation, heading offset and ground plane. Not yet wired into the tracker, which still reads Gazebo truth.
- `scene_consistency.py`: `validate_camera_info` (ROS CameraInfo-like dict vs config intrinsics), `check_scene_consistency` (config vs sdf cameras, target cylinder, launch spawns); module docstring holds the live rendered-image validation protocol (not run: no Gazebo).
- Tests: `test_homing_lateral_sign.py` (ENU/FLU oracle, original-code RED), `test_homing_gazebo_camera_model.py` (projection/reconstruction, PinholeCamera rendering in ENU/FLU), `test_homing_scene_consistency.py`, `test_homing_origin_alignment.py`.
- Fix round 1: once locked, the Gazebo (mounts) path froze velocity/acceleration forever as soon as height dropped to/below the 1.5 m visible-centre reference height (`reference_relative_position` returns None there, not only when the ray points above the horizon). `HomingGuidance._sightline_and_relative` now falls back to steering from the camera-ray bearing alone in that case, keeping `tracking=True`; the above-horizon case is unchanged (still a hold). `_MIN_COURSE_SPEED_M_S` is now also a `HomingGuidance(min_course_speed_m_s=...)` parameter, and `homing_config.PROPOSED_MIN_COURSE_SPEED_M_S` (1.0, unchanged value) feeds it from `guidance_from_config`.

## 4.4.0 - Homing diagnostics, run outcomes and replay (plan Phase 1)
- `guidance/telemetry.py`: per-run JSONL (`Results/gazebo_runs/<unique run id>/telemetry.jsonl`, never overwritten) with a metadata record (config, firmware id, PX4 params, rendering mode, timing), per-tick `TickRecord`, events (acquisition/handoff/loss) and a final outcome. `FrameStore` saves lossless uint8 `.npy` frames (around events, or all with `HOMING_SAVE_ALL_FRAMES=1`). Every write is failure-tolerant and a repeatedly failing log disables itself.
- `outcome.py`: `RunOutcome` (TARGET_REACHED, GROUND_MISS, TIMEOUT, TARGET_LOST, ESTIMATOR/OFFBOARD/SIMULATOR/STARTUP_FAILURE) and `classify_run`; ground contact alone is never TARGET_REACHED. `PROPOSED_REACH_RADIUS_M` (1.5 m) in `homing_config.py` is an unverified proposal.
- `guidance/replay.py`: `replay_run` reruns guidance on recorded inputs and frames without Gazebo and compares decisions; it stops at the first tick with missing frames. `guidance_from_config` is now shared by the tracker and replay.
- `Dockerfiles/camera_homing_tracker.py` records each step and the scored outcome through thin, exception-guarded calls. Not flown: PX4 params and firmware id come from the environment (`PX4_FIRMWARE_ID`), not queried live.

## 4.3.0 - Robust target detection (plan Phase 4)
- `guidance/pixels.py`: `detect_target` finds the red target as a connected component (red dominance and saturation, area/fill/aspect checks, border clipping flagged with inflated uncertainty) and returns a frozen `TargetObservation` with `DetectionStatus` NO_FRAME/NO_DETECTION/AMBIGUOUS/DETECTED.
- `TrackPrediction` gates candidates; `TargetAssociator` keeps a per-camera track so distractors cannot move the reference. Not yet wired into `homing.py`; the legacy `target_*` functions are unchanged.

## 4.2.1 - Camera homing documentation
- Correct the current Python controller selection and distinguish it from Gazebo's PX4 control.
- Document Gazebo's GPU and software rendering, startup, automatic stop, and recording behavior.

## 4.2.0 - Stop Gazebo after homing
- Save the flight page and stop the Gazebo session when homing reaches the ground or its configured timeout.
- Supervise and clean up only the processes started by the simulation launcher.

## 4.1.0 - NVIDIA Gazebo rendering
- Use the NVIDIA GPU for Gazebo cameras and visualization by default, verifying the OpenGL renderer in the container.
- `LIBGL_ALWAYS_SOFTWARE=1` explicitly selects software rendering; headless camera rendering retains the X display.

## 4.0.0 - Root Gazebo homing launcher
- Moved `run_camera_homing_sitl.sh` to the project root; the old `Dockerfiles/` path no longer works.

## 3.14.2 - Gazebo motor startup before homing
- Disable throw launch, wait for the motors to spool, and stabilize at the starting altitude before commanding camera homing.
- Start the Gazebo flight recording after the startup hover so initialization does not consume the flight timeout.

## 3.14.1 - Stable Gazebo homing startup
- Pause physics while spawning and disable gravity on every Hummingbird link, including the GPS, until PX4 arms in offboard mode.
- Retain the last PX4 heartbeat state between updates instead of repeatedly requesting arm and offboard.

## 3.14.0 - Larger Gazebo homing target
- Tripled the Gazebo target cylinder's dimensions to 1.5 m diameter and 3 m height, keeping it on the ground.

## 3.13.2 - Local Conda environment
- Created the Python 3.11 `rotorpy` environment with an editable installation of all project extras and CPU-only PyTorch.
- Selected it in VS Code, configured the Conda executable, and documented shell activation without relying on Conda being on PATH.

## 3.13.1 - Forward camera clear of the airframe
- The Gazebo forward camera sits 0.36 m ahead of the body and 0.05 m up, still pitched 10° down. The shell, legs, and rotors are behind that lens.
- Propeller visuals are scaled to the 0.1 m rotor disk. The mesh was a metre across, which is the red and blue shape in the forward picture.

## 3.13.0 - Gazebo homing page
- The Gazebo tracker writes `Results/camera_homing_gazebo.html` when the flight hits the ground, times out, or the tracker stops. The page is the same slider view as `Results/camera_homing.html`.
- The SITL container installs Pillow when it is missing, because that page encodes the camera pictures with it.

## 3.12.6 - Restart Gazebo if spawn aborts it
- The Hummingbird spawn waits until Gazebo is serving, then checks that `gzserver` is still alive. An empty transport header aborts the server during spawn; the launch starts Gazebo again instead of booting PX4 against a dead server.

## 3.12.5 - Kill the sim container first
- `run_hummingbird_sitl.sh` removes `rotorpy-hummingbird-sitl` before checks, xhost, or starting Gazebo.

## 3.12.4 - Software GL for the Gazebo window
- The homing launch uses software OpenGL. Nouveau aborts `gzclient` after the window opens, with an empty Gazebo header. `LIBGL_ALWAYS_SOFTWARE=0` keeps the GPU.

## 3.12.3 - Gazebo server survives startup on this host
- The SITL container is not started with a docker TTY. That TTY makes `gzserver` abort with an empty transport header.
- If `gzserver` still aborts, the next attempt clears it and uses software GL. Nouveau DRI was aborting the server the same way.

## 3.12.2 - Retry PX4 arm for camera homing
- Offboard is requested as PX4 main mode 6. The packed heartbeat value was cast away, so the vehicle stayed in loiter.
- The tracker asks again until the heartbeat is armed in offboard. Gravity stays off until then, so the 30 m spawn does not fall during the wait.
- The offboard setpoint is velocity and yaw. The homing acceleration is the RotorPy feedforward, and PX4 was tilting the Hummingbird with it.
- The airframe enables in-air start and sets hover thrust for the 0.68 kg vehicle. PX4 otherwise treats the 30 m spawn as landed and commands zero thrust.

## 3.12.1 - Gazebo path variables on a fresh container
- `sitl_hummingbird_run.sh` sets `GAZEBO_PLUGIN_PATH`, `GAZEBO_MODEL_PATH`, and `LD_LIBRARY_PATH` before sourcing PX4's setup script. Nounset was aborting the launch when those were unset.

## 3.12.0 - Gazebo camera homing
- The Hummingbird Gazebo model has the forward and upward cameras from `examples/camera_homing.jsonc`.
- `Dockerfiles/run_camera_homing_sitl.sh` spawns a red 0.5 m by 1 m cylinder on the homing ground point and runs `camera_homing_tracker.py`, which homes from those pictures through PX4 offboard.

## 3.11.0 - Hummingbird Gazebo Classic injection
- `gazeboModels/hummingbird/` is the AscTec Hummingbird mesh and a PX4 Gazebo Classic SDF (mass 0.68 kg, plus layout, arm 0.17 m).
- `Dockerfiles/run_hummingbird_sitl.sh` bind-mounts that model into the existing image `px4-noetic-sim-ros`. `sitl_hummingbird_run.sh` spawns it. `px4_noetic_sim_bash.sh` opens a shell with the same mounts.

## 3.10.0 - Camera homing JSONC
- `examples/camera_homing.jsonc` holds the homing scene, positions, and controller. `controller` is `geometric` (SE3Control) or `pid` (Px4Cascade). The shipped choice is `pid`.
- `examples/camera_homing.py` reads that file for the flight and the sweep.

## 3.9.0 - One folder per controller
- `SE3Control`, `Px4Cascade`, `RaptorFoundationPolicy`, and the control template each live in their own package under `rotorpy/controllers/`.
- Class imports stay `rotorpy.controllers.quadrotor_control`, `rotorpy.controllers.px4_cascade`, and `rotorpy.controllers.raptor`. PX4 gains and math import from `rotorpy.controllers.px4_cascade`.

## 3.8.0 - PX4 cascade on homing
- `Px4Cascade` runs the PX4 position, velocity, attitude, and rate loops and outputs Hummingbird motor speeds.
- The homing runner commands the velocity tier in PX4 NED. Firmware gains are unchanged.

## 3.7.1 - Upright homing acceleration
- The flat output accelerates along the homing velocity miss over half a second, scaled so the specific force stays inside a 45° upright cone. The opening motion is a pitch on the short arc.

## 3.7.0 - Upward camera for the homing close
- A second camera, aimed 70° above the nose, steers once it sees the target. Homing aims through the center of the square around that target.
- The flight stops at the ground or at 20 seconds. The homing page shows both pictures, that square, and both frustums.

## 3.6.1 - One body frame on the homing page
- The camera drawing is the frustum only. The body FRD axes are the only coordinate frame, beside the drone sketch.

## 3.6.0 - FRD drone and camera frustum
- The homing-page body axes and the X airframe are forward-right-down. Prop circles stay on the upper side, opposite the down axis.
- The camera frustum and camera triad are drawn again, in the simulator camera frame.

## 3.5.3 - Homing scene without the frustum
- The Plotly scene no longer draws the camera frustum or the camera triad. The body axes and the drone remain.

## 3.5.2 - Larger drone glyph
- The homing-page drone is twice the 3.5.1 size. Prop circles stay just above the arm tips.

## 3.5.1 - Smaller drone glyph
- The homing-page drone is one tenth the previous size. Prop circles stay just above the arm tips.

## 3.5.0 - Drone glyph on the homing page
- The Plotly scene draws the Hummingbird as an X. Forward arms and their prop circles are green, rear arms and their prop circles are red.
- Each prop circle sits a short distance above its arm tip, on the body-up side.

## 3.4.0 - Camera homing Plotly page
- `examples/camera_homing.py` writes `Results/camera_homing.html` instead of opening the matplotlib animation.
- The page shows the 3D path, target box, body axes, and camera frustum beside that instant's camera image. The slider selects the capture time.

## 3.3.0 - FRD-NED world frame
- The world switch is `World.frame`: a world chooses FLU-ENU or FRD-NED, and an unset frame stays FLU-ENU. Gravity, thrust, the IMU, wind filters, and the PX4 and ArduPilot bridges follow that choice.
- `rotorpy/frames.py` converts vectors, attitudes, yaw, bounds, and flat outputs between FLU-ENU and FRD-NED, and `apply_frame` calls `set_frame` on objects that define it.
- Example equivalence: the listed examples finish on the same physical state in FRD-NED as in FLU-ENU after those maps.

## 3.2.0 - Camera homing animation
- `examples/camera_homing.py` plays the nominal flight as a third-person 3D view beside the drone camera, using the same animation layout as `camera_demo.py`.
- The sweep still prints one line per navigation constant.

## 3.1.0 - Camera homing example
- `examples/camera_homing.py` prints one nominal camera-homing miss or timeout.
- `--sweep` repeats the flight for navigation constants 2, 3, 4, and 5.

## 3.0.2 - Writable acceleration array in wind filter example
- `examples/wind_filter_example.py` copies the acceleration array before subtracting thrust. Pandas 3 `to_numpy()` is read-only.

## 3.0.1 - NumPy 2 waypoint stacking
- `examples/multimav_multiprocessing_demo.py` uses `np.vstack` for MinSnap waypoints. `np.row_stack` was removed in NumPy 2.
