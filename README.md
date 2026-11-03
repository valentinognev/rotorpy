# RotorPy
A Python-based multirotor simulation environment with aerodynamic wrenches, useful for education and research in estimation, planning, and control for UAVs.
<p align="center"><img src="/media/double_pillar.gif" width="32%"/><img src="/media/gusty.gif" width="32%"/><img src="/media/minsnap.gif" width="32%"/></p>

**NEW in `v3.0`**: RotorPy now includes a fast (non-photorealistic) camera sensor for developing and testing vision algorithms (visual odometry, vision-based policies, etc.). Check out the new examples: [`camera_demo.py`](/examples/camera_demo.py), [`camera_visualization.py`](/examples/camera_visualization.png), [`visual_servoing_policy.py`](/examples/visual_servoing_policy.py), and [`camera_homing.py`](/examples/camera_homing.py).
<p align="center"><img src="/media/camera_demo.gif" width="50%"/><img src="/media/visual_servoing_example.gif" width="50%">
<p align="center"><img src="/media/camera_highlight.png" width="100%">

World blocks can now carry visual features: 3D points with RGB colors and optional descriptor vectors. The new `PinholeCamera` sensor projects those features into an image using the standard pinhole + distortion model. `BatchedPinholeCamera` renders images for many UAVs in parallel. 

## Purpose and Model Scope
The original focus of this simulator was on accurately simulating rotary-wing UAV dynamics using lumped parameter representations of the aerodynamics, primarily for graduate level course design and exploratory research. These aerodynamic effects, outlined in [`rotorpy/vehicles/README.md`](/rotorpy/vehicles/README.md), are negligible at hover in still air; however, as relative airspeed increases (e.g. for aggressive maneuvers or in the presence of high winds), they quickly become noticeable and force the student/researcher to reconcile with them. 

As RotorPy continues to grow, the focus is now on building a sufficiently realistic dynamics simulator that can also scale to quickly generate extremely large datasets as a tool for deep learning, reinforcement learning, and Monte Carlo studies on existing (or new!) algorithms in estimation, planning, and control. 

The engine is designed from the bottom up to be lightweight, easy to install with very limited dependencies or requirements, and interpretable to anyone with basic working knowledge of Python. The intent is for users to gain intuition about UAV dynamics and learn how to develop control and/or estimation algorithms for rotary wing vehicles especially in the presence of aerodynamic wrenches and strong winds. 

We hope that this repository will be a helpful resource for educators and researchers. 

# Installation

RotorPy can be installed using `pip`:

```bash
pip install rotorpy 
```

This will install the minimum packages to run the most basic version of the simulator. But to get the most of RotorPy you may want to consider...

```bash
pip install rotorpy.[all]       # the complete package, includes everything below too
pip install rotorpy.[testing]   # if you're planning on developing and need to run the testing suite
pip install rotorpy.[batched]   # for the batched (parallelized) environments
pip install rotorpy.[learning]  # for learning with the gymnasium environment
pip install rotorpy.[px4]       # for integration with PX4
```

For other tagged versions, see `pyproject.toml`. It's an evolving thing.

## Local Conda Environment

For this checkout, use the `rotorpy` environment with Python 3.11:

```bash
source "$HOME/Anaconda3/etc/profile.d/conda.sh"
conda activate rotorpy
python examples/camera_homing.py
```

To recreate the environment, run `conda create -n rotorpy python=3.11 pip -y`,
activate it, and run `python -m pip install -e '.[all]'` from the project root.
The current installation uses CPU-only PyTorch, installed before the project
with `python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`.
The workspace selects this environment for Python and debugging.

# Usage

There are a few example scripts found in `rotorpy/examples/` that demonstrate how to use RotorPy in a variety of ways including for Monte Carlo evaluations, reinforcement learning, swarms, and even visual servoing (as of `v3.0`). 

#### Regular usage
A good place to start would be to reference the `rotorpy/examples/basic_usage.py` script. It goes through the necessary imports and how to create and execute an instance of the base simulation environment. 
 
At minimum the simulator requires vehicle, controller, and trajectory objects. The vehicle (and potentially the controller) is parameterized by a unique parameter file, such as `rotorpy/vehicles/hummingbird_params.py`. There is also the option to specify your own IMU, world bounds, and how long you would like to run the simulator for. A `World` chooses FLU-ENU or FRD-NED, and unset stays FLU-ENU. 

The output of the simulator is a dictionary containing a time vector and the time histories of all the vehicle's states, inputs, and measurements.

Below is a minimum working example: 

```python
import numpy as np
from rotorpy.environments import Environment
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.crazyflie_params import quad_params
from rotorpy.controllers.quadrotor_control import SE3Control
from rotorpy.trajectories.lissajous_traj import TwoDLissajous
from rotorpy.wind.default_winds import SinusoidWind

sim_instance = Environment(vehicle=Multirotor(quad_params),           # vehicle object, must be specified. 
                           controller=SE3Control(quad_params),        # controller object, must be specified.
                           trajectory=TwoDLissajous(),                # trajectory object, must be specified.
                           wind_profile=SinusoidWind(),               # OPTIONAL: wind profile object, if none is supplied it will choose no wind. 
                           sim_rate     = 100,                        # OPTIONAL: The update frequency of the simulator in Hz. Default is 100 Hz.
                           camera       = None,                       # OPTIONAL: camera sensor object, by default won't use the camera if none is supplied.
                           imu          = None,                       # OPTIONAL: imu sensor object, if none is supplied it will choose a default IMU sensor.
                           mocap        = None,                       # OPTIONAL: mocap sensor object, if none is supplied it will choose a default mocap.  
                           estimator    = None,                       # OPTIONAL: estimator object
                           world        = None,                       # OPTIONAL: the world, same name as the file in rotorpy/worlds/, default (None) is empty world
                           safety_margin= 0.25                        # OPTIONAL: defines the radius (in meters) of the sphere used for collision checking
                       )

x0 = {'x': np.array([0,0,0]),
      'v': np.zeros(3,),
      'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
      'w': np.zeros(3,),
      'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
      'rotor_speeds': np.array([1788.53, 1788.53, 1788.53, 1788.53])}
sim_instance.vehicle.initial_state = x0

# The results are a dictionary containing the relevant state, input, and measurements vs time.
results = sim_instance.run(t_final      = 20,       # The maximum duration of the environment in seconds
                           use_mocap    = False,       # Boolean: determines if the controller should use the motion capture estimates. 
                           terminate    = False,       # Boolean: if this is true, the simulator will terminate when it reaches the last waypoint.
                           plot            = True,     # Boolean: plots the vehicle states and commands   
                           plot_mocap      = True,     # Boolean: plots the motion capture pose and twist measurements
                           plot_estimator  = True,     # Boolean: plots the estimator filter states and covariance diagonal elements
                           plot_imu        = True,     # Boolean: plots the IMU measurements
                           animate_bool    = True,     # Boolean: determines if the animation of vehicle state will play. 
                           animate_wind    = True,    # Boolean: determines if the animation will include a scaled wind vector to indicate the local wind acting on the UAV. 
                           verbose         = True,     # Boolean: will print statistics regarding the simulation. 
                           fname   = None # Filename is specified if you want to save the animation. The save location is rotorpy/data_out/. 
                    )

```
#### Batched Simulations
RotorPy includes a batched environment which can simulate multiple drones in parallel on CPU or GPU. For simulations of >1000 drones, we have observed speedups of 25x purely on CPU compared to simulating all drones sequentially. We have also implemented batched versions of existing control, trajectory, sensor, and wind classes. See [`examples/batched_simulation.py`](/examples/batched_simulation.py) for how to use the batched simulation, and [`examples/benchmark_batched_simulation.py`](/examples/benchmark_batched_simulation.py) to measure the speedup on your own system.

#### Reinforcement Learning
RotorPy includes a custom Gymnasium environment, `QuadrotorEnv`, which is a stripped down version of the regular simulation environment intended for applications in reinforcement learning. `QuadrotorEnv` features all the aerodynamics and motor dynamics, but also supports different control abstractions ranging from high level velocity vector commands all the way down to direct individual motor speed commands. This environment also allows the user to specify their own reward function. 

For an example of how to interface with this environment, see `rotorpy/examples/gymnasium_basic_usage.py`. You can also see an example of training a quadrotor to hover using this environment in `rotorpy/examples/ppo_hover_train.py` and `rotorpy/examples/ppo_hover_eval.py`. 

You can find this new environment in the `rotorpy/learning/` module. 

#### Camera homing
[`examples/camera_homing.py`](/examples/camera_homing.py) flies a Hummingbird at a red box and writes `Results/camera_homing.html`. The start, the target, both cameras, the timing, and the inner loop are [`examples/camera_homing.jsonc`](/examples/camera_homing.jsonc). The Python simulation uses `geometric` (`SE3Control`) in the current configuration; `pid` selects `Px4Cascade`. `--sweep` repeats the flight for the navigation constants in that file.

For Gazebo, run `./run_camera_homing_sitl.sh` from the project root. It requires
Docker, the existing `px4-noetic-sim-ros` image, and a working X display. The
launcher replaces any existing `rotorpy-hummingbird-sitl` container and
temporarily disables X server access control.

Gazebo's cameras and visualization use the NVIDIA GPU by default, requiring an
NVIDIA driver and Docker's NVIDIA runtime; physics may still run on the CPU.
Run `LIBGL_ALWAYS_SOFTWARE=1 ./run_camera_homing_sitl.sh` to use software
rendering. `HEADLESS=1` hides the Gazebo window but still requires a display for
the camera sensors.

Gazebo uses PX4 offboard control regardless of the Python `controller` choice.
After motor startup and a stable hover, it flies until ground contact or
`timeout_s` (20 seconds in the current configuration), then saves
`Results/camera_homing_gazebo.html` if camera frames are available and stops.

#### And much more
RotorPy is intended to be flexible to support the aerial robotics field as it expands. Explore the examples in this repository and [citations](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17472125763442225756) to see the many ways RotorPy can be used to learn more about aerial robotics. 

# Development

It is rather straightforward if you would like to add more tracking methods into the simulator. For instance, if you'd like to add a new trajectory generator or a new controller, we've added respective templates that you can use under `rotorpy/trajectories/` and `rotorpy/controllers/`. Each controller is its own package there: `quadrotor_control/` (`SE3Control`), `px4_cascade/` (`Px4Cascade`), `raptor/`, and `controller_template/`. If you'd like to add your own wind field, you can add a new class in `rotorpy/wind/` following the template there. 

As for adding more core functionality (e.g., sensors, new vehicle dynamics, animations, etc.), those require a bit more effort to make sure that all the inputs and outputs are set up accordingly. One piece of advice is that the main loop occurs in `rotorpy/simulate.py`. Under the `while` loop, you can see the process by which the vehicle dynamics, trajectory generator, IMU sensor, and controller interface with each other. 


# Citation

If you use RotorPy for your work please cite our companion workshop paper contributed to the [RS4UAVs Workshop at ICRA 2023](https://imrclab.github.io/workshop-uav-sims-icra2023/): 

```
@article{folk2023rotorpy,
  title={{RotorPy}: A Python-based Multirotor Simulator with Aerodynamics for Education and Research},
  author={Folk, Spencer and Paulos, James and Kumar, Vijay},
  journal={arXiv preprint arXiv:2306.04485},
  year={2023}
}
```

See how academics have been using RotorPy: 
[![Citations](https://img.shields.io/badge/Citations-Google%20Scholar-green?logo=googlescholar)](https://scholar.google.com/scholar?oi=bibs&hl=en&cites=17472125763442225756)

# Acknowledgements

We would like to acknowledge [Jimmy Paulos](https://github.com/jpaulos) who wrote the majority of the underlying engine for RotorPy, and the teaching assistants who contributed code to the initial version of this simulator, especially [Dan Mox](https://github.com/danmox), [Laura Jarin-Lipschitz](https://github.com/ljarin), [Rebecca Li](https://github.com/rebeccali), [Shane Rozen-Levy](https://github.com/ShaneRozenLevy), [Xu Liu](https://github.com/XuRobotics), [Yuezhan Tao](https://github.com/tyuezhan), [Yu-Ming Chen](https://github.com/yminchen), and [Fernando Cladera](https://github.com/fcladera).

## Architecture
Each controller is a package under `rotorpy/controllers/`. `Px4Cascade` in `rotorpy/controllers/px4_cascade/` outputs Hummingbird `cmd_motor_speeds` from the PX4 position, velocity, attitude, and rate loops. For the Python camera-homing simulation, `examples/camera_homing.jsonc` selects that cascade with `"controller": "pid"`, or `SE3Control` from `rotorpy/controllers/quadrotor_control/` with `"controller": "geometric"`. Other examples keep `SE3Control`.

`gazeboModels/hummingbird/` is an AscTec Hummingbird for PX4 Gazebo Classic. The forward camera is 0.36 m ahead of the body and 0.05 m up, pitched 10° down, so the airframe stays outside that picture. Propeller visuals are scaled to the 0.1 m rotor disk. The upward camera is at the body origin, pitched 70° above the nose. `Dockerfiles/run_hummingbird_sitl.sh` starts it in the existing image `px4-noetic-sim-ros` by bind-mounting that model, `Dockerfiles/sitl_hummingbird_run.sh`, and airframe `1100_gazebo-classic_hummingbird`. `Dockerfiles/px4_noetic_sim_bash.sh` opens a shell in the same image with those mounts. The container name is `rotorpy-hummingbird-sitl`, so it does not remove a container already named `px4-noetic-sim-ros`.

`run_camera_homing_sitl.sh` places that Hummingbird at the camera-homing start, 30 m up, aimed at a red cylinder on the homing ground point (diameter 1.5 m, height 3 m). `Dockerfiles/camera_homing_tracker.py` runs the `guidance/startup.py` `StartupGate` (fresh PX4 estimate, valid attitude, confirmed armed+offboard, motor spin in the airborne-spawn mode, and a sustained stable hover, every wait bounded) before starting the homing law on the two camera topics and commanding PX4 offboard velocity. The tracker is a thin ROS/MAVLink shell: per-camera buffering (`guidance/observation_buffer.py`), timestamp-aware PX4 vehicle state (`guidance/vehicle_state.py`, attitude history + clock-pause/reset/jump detection), independent periodic setpoint streaming (`guidance/setpoint_stream.py`), off-control-path recording (`guidance/recording_worker.py`) and comms watchdogs (`guidance/watchdog.py`) are importable, unit-tested pure modules; Gazebo's own model state feeds only recording and outcome scoring, never guidance or the startup/arming state machine. The setpoint stream's documented failure response (expiry or a fatal step/watchdog failure) is `PROPOSED_FAILURE_COMMAND = None`: send nothing and let PX4's own configured failsafe (RTL/land/hold) take over, rather than this tracker fabricating a velocity command of its own. It writes `Results/camera_homing_gazebo.html`, the same page as the RotorPy flight.

`HomingGuidance` (`guidance/homing.py`) drives per-camera visibility/handoff through an explicit, recoverable state machine (`guidance/tracking.py`): `STARTUP -> ACQUIRE -> TRACK -> TEMPORARY_LOSS -> REACQUIRE`, with terminal `ABORT` (the guidance's own one-way failure response to a prolonged loss or runaway per-camera track uncertainty, both bounded, named `PROPOSED_*` thresholds in `guidance/homing_config.py`) and `COMPLETE` (set only by an external `HomingGuidance.complete(reason)` call from the truth-side scorer/runner; guidance itself never reads truth). It replaced a one-way "upward camera latches, then any later loss is permanent" bug shared by both the legacy no-mounts path and the Gazebo mounts path. `HomingCommand.tracking_state`/`abort_reason` expose the state to callers; the documented, configurable decision is that `ABORT` maps to the T3 `SetpointStream.mark_fatal()` failure-response path and the T1 `RunOutcome.TARGET_LOST` outcome (wired in `camera_homing_tracker.py`'s `step()`), never a silently-held stale command.

On the Gazebo (mounts) path only, `guidance/target_estimator.py`'s `TargetMotionEstimator` is a constant-velocity Kalman filter over the target's own ground velocity: it turns each tracked camera ray (plus its pixel uncertainty, clip flag, and confidence from `pixels.TargetObservation`) into a ground-plane position fix, with the *known* vehicle velocity entering the predict step as an exact control input so a maneuvering drone's own apparent sightline drift cancels out instead of being mistaken for target motion. Near-horizon or severely clipped rays are caught by a numeric ray-perturbation Jacobian and flagged unobservable rather than producing an exact position; a Mahalanobis gate rejects outliers; `predict_to` compensates processing delay and reports relative velocity/closing speed. `HomingCommand.target_motion` carries the resulting `TargetMotionEstimate` (`status` `OBSERVABLE`/`UNCERTAIN`/`UNOBSERVABLE`) into telemetry (`telemetry.target_motion_fields_from_command`).

Still Gazebo (mounts) path only, `HomingGuidance._steer` (`guidance/homing.py`) now consumes that estimate: when `status is OBSERVABLE`, `guidance/navigation.py`'s `pn_lead_point` folds proportional navigation into the aim point itself (`relative + nav_constant * lead_time_s * relative_velocity`) so changing `nav_constant` demonstrably changes the transmitted velocity even though acceleration fields stay masked by default; otherwise steering falls back to the pre-Phase-7 bearing-only aim point and records why (`"target_motion_unavailable:<reason>"`). `guidance/speed.py`'s `max_feasible_velocity` replaces the per-component `desired_velocity` cap with a total-speed (norm) objective -- commanding the vehicle's full configured speed toward the aim point, not a direction-dependent one. `guidance/command_limits.py`'s `limit_command` then bounds that request to a `VehicleEnvelope` (speed, vertical rate, acceleration magnitude, jerk, yaw rate, all named `PROPOSED_*`/unverified in `homing_config.vehicle_envelope()`) plus a visibility margin that scales both the speed and yaw-rate caps down as the tracked target nears the image edge, and a continuity bound on the per-tick velocity/acceleration/yaw step; every bound actually applied is a named reason. `HomingCommand.requested_velocity_ned`/`limiting_reasons` (and `telemetry.setpoint_fields_from_command`) expose the pre-limit request alongside the limited/transmitted one, `None` on the legacy no-mounts path (which keeps its exact unlimited `desired_velocity`/`pn_acceleration` output). `HomingGuidance.enable_acceleration_feedforward` (default `False`) is a documented, tested, off-by-default flag the tracker's `_send` mask reads; flipping it is not itself "the fix" (project history: accel feedforward was unstable) and requires the live-flight validation this offline environment cannot perform.

`HomingGuidance.update`'s `t` argument is now the image's own *capture* time (every detection/dt/estimator-update computation lines up with the vehicle state `VehicleState.sample_at` already drew at that same instant); the separate, later `control_t` (defaulting to `t`) is the processing instant the command actually takes effect, and only `TargetMotionEstimator.predict_to` advances to it, compensating for exactly the capture-to-command processing delay. `Dockerfiles/camera_homing_tracker.py` passes the image's capture time (falling back to the processing instant before any frame has ever arrived) as `t` and `rospy.get_time()` as `control_t`.

Startup, outcomes and replay (plan Phase 8). `guidance/startup.py`'s `StartupGate` is the pure state machine (`WAIT_ESTIMATE -> ENGAGE -> [MOTOR_START | CLIMB] -> STABILIZE -> HOMING`, terminal `FAILED`) with named `PROPOSED_*` bounds in `guidance/homing_config.py` (`startup_thresholds()`); every timeout is a `STARTUP_FAILURE` outcome. Its stream policy is explicit and tested: nothing is sent without a fresh PX4 estimate, from the moment it is fresh the tracker restamps a startup setpoint (a velocity hold plus the vehicle's own heading) every step -- independent of when the first camera image arrives -- and the homing command is transmitted only after the gate opens (handed over at its own age so a stale command expires). Startup mode is an explicit flag (`--startup-mode`, env `HOMING_STARTUP_MODE` in the launcher): `takeoff` (spawn on the ground, arm, climb to the start height; preferred, but only exercised offline as a state machine and with the estimator origin moved to the ground spawn) or `airborne_spawn` (legacy simulation mode: spawn at the start height with gravity disabled until the rotors spin; still the proposed default only because the scene and every prior flight use it -- flip `homing_config.PROPOSED_STARTUP_MODE` after live validation). `guidance/run_failure.py` splits errors into `RecoverableError` (skipped, counted against a bounded `RecoverableBudget`) and fatal `FatalError`s (and anything unexpected): fatal latches the setpoint stream and ends the run. `guidance/run_lifecycle.py`'s `RunLifecycle` writes exactly one outcome per run (also announced as a `RUN_OUTCOME` stdout line, so a run without telemetry or with a failing disk still ends unambiguously; a teardown with no outcome becomes `SIMULATOR_FAILURE`), and flushes/closes the recorder on every termination path. `guidance/scoring.py` is the truth-side module (no guidance/estimator module imports it: `tests/test_homing_truth_boundary.py`): `TruthScorer` interpolates target and vehicle truth to common sim times, catches contact between samples with exact segment-vs-cylinder closest-approach tests (plus a collision-event hook), keeps target contact separate from ground contact, and feeds `outcome.classify_run` -- ground contact alone is never `TARGET_REACHED`. Termination comes from the scorer, not the recorder or any camera. `HomingRecord.geometry` (`ReplayGeometry`) makes the replay page draw the actual target cylinder (moving if sampled) and list the real camera mounts, pitches and FOV from the config.

`Dockerfiles/px4_characterization.py` is a documented, importable (not run here) PX4 parameter-dump and command-response-characterization script; its `analyze_step_response`/`analyze_velocity_tracking` arithmetic is unit-tested on synthetic data so a future live-flight log only has to supply the recording.

## Reading order for agents
1. Read this `README.md`.
2. Read `UPDATES.md` for the change history and current state before working. 
