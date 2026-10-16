# FRD-NED Frame Convention Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A world can be created in FRD-NED as well as the current FLU-ENU, gravity and the controller follow that choice, and the listed examples finish on the same physical state.

**Architecture:** `rotorpy/frames.py` holds the maps and `apply_frame`. `World.frame` is the switch. `Environment`, `simulate`, and `simulate_batch` call `set_frame` on the vehicle, controller, IMU, and estimator after those objects already exist. Parameter files stay authored in FLU. Autopilot bridges keep today's FLU-ENU packing and skip the axis rewrite when the world is already FRD-NED.

**Tech Stack:** Python >= 3.8, NumPy, SciPy `Rotation`, pytest, existing RotorPy `Multirotor`, `SE3Control`, `Imu`, `World`.

**Spec:** `docs/superpowers/specs/2026-10-16-frd-ned-frame-design.md`

## Global Constraints

- Python >= 3.8. Type hints use `typing.Optional` and `typing.Tuple`, not the `|` union syntax.
- Do not modify `rotorpy/guidance/frames.py`, vehicle parameter files, the gym environment, RAPTOR, visual servoing, or camera homing.
- Do not git commit or git push. The user has not asked for commits.
- `g = 9.81` in dynamics, control, IMU, and wind filters. PX4 milli-g stays `9.80665`.
- Unset frame is `FrameConvention.FLU_ENU`. Quaternions stay scalar-last `[x, y, z, w]` inside the simulator.
- `set_frame` is idempotent. Authored FLU geometry is stored and the active geometry is derived from it.
- Equivalence tolerances: position `1e-2` m, velocity `1e-2` m/s, attitude `1e-2` rad, body rate `1e-2` rad/s. Fitted drag `1e-3`. Wind RMSE and final body-wind `1e-2` m/s. Camera keypoints within 2 pixels.
- Except the wind-filter script, equivalence IMUs use zero noise density and zero random walk. The wind-filter script keeps the default noisy IMU.
- Plots and animation stay off in tests. Do not open a MAVLink or ArduPilot connection.
- Default pytest stays `not run_examples and not homing_e2e and not frame_equivalence and not frame_wind`.

## File structure

- `rotorpy/frames.py` — enum, maps, AABB and flat-output conversion, `apply_frame`.
- `rotorpy/world.py` — `frame` on `World`.
- `rotorpy/vehicles/multirotor.py` — `set_frame` on `Multirotor` and `BatchedMultirotor`.
- `rotorpy/controllers/quadrotor_control.py` — `set_frame` on `SE3Control` and `BatchedSE3Control`.
- `rotorpy/sensors/imu.py` — `set_frame` on `Imu` and `BatchedImu`.
- `rotorpy/estimators/wind_ukf.py` and `wind_ekf.py` — `set_frame`.
- `rotorpy/environments.py` and `rotorpy/simulate.py` — apply `world.frame`.
- `rotorpy/vehicles/px4_multirotor.py` and `ardupilot_multirotor.py` — frame-dependent packing helpers.
- `examples/multimav_multiprocessing_demo.py` — `run_sim` takes the world frame.
- `examples/wind_filter_example.py` — thrust removal along `thrust_axis`.
- `tests/test_frames.py`, `test_world_frame.py`, `test_frame_vehicle.py`, `test_frame_control.py`, `test_frame_imu.py`, `test_frame_batched.py`, `test_frame_wind_model.py`, `test_frame_propagate.py`, `test_frame_bridges.py`, `test_frame_equivalence.py`, `test_frame_wind_trials.py`.
- `UPDATES.md` — version `3.3.0`. `README.md` — one sentence that a world chooses FLU-ENU or FRD-NED.

## Parallel execution

Wave 1: Task 1.

Wave 2, together after wave 1: Tasks 2, 3, 5, 6, 8.

Wave 3, together after their wave-2 dependencies: Task 4 after Task 3. Task 7 after Tasks 3, 5, and 6.

Wave 4, together after wave 3: Task 9 after Tasks 2, 4, 5, 6, 7, and 8. Tasks 11 and 12 after Tasks 1 and 3. Task 10 after Tasks 3 and 5. Task 14 after Task 3.

Wave 5: Task 13 after Tasks 9 and 10. Task 15 after Tasks 8, 9, and 14.

Wave 6: Task 16 after wave 5.

A reviewer checks a task before a later wave imports it. Reviewers for tasks in the same wave can run together.

## Review Focus

- `set_frame(FRD_NED)` twice must leave rotor y negated once. Tested in Task 3.
- An FRD hover force along world −z must command the identity quaternion, with positive collective thrust. Tested in Task 5.
- ENU yaw 0 converts to NED yaw `π/2`. Leaving both yaw commands at 0 points the nose on different compass headings. Tested in Task 1 and used by Task 13.
- FRD PX4 acceleration stays in milli-g. Tested in Task 11.
- The wind-filter FRD run reuses the seed-0 parameter draws and does not roll them again. Tested in Task 15.

---

### Task 1: Frame maps

**Files:**
- Create: `rotorpy/frames.py`
- Test: `tests/test_frames.py`

**Interfaces:**
- Consumes: nothing
- Produces: `FrameConvention` with `FLU_ENU = "flu_enu"` and `FRD_NED = "frd_ned"`. `M_ENU_TO_NED` and `M_FLU_TO_FRD` as `(3, 3)` arrays. `enu_to_ned`, `ned_to_enu`, `flu_to_frd`, `frd_to_flu` each `(3,) -> (3,)`, copying the input. `attitude_enu_flu_to_ned_frd(q_xyzw) -> (4,)` and `attitude_ned_frd_to_enu_flu(q_xyzw) -> (4,)`. `yaw_enu_to_ned(yaw) -> float`, `yaw_ned_to_enu(yaw) -> float`, `yaw_rate_enu_to_ned(yaw_dot) -> float` (`yaw_dot_ned = -yaw_dot_enu`). `aabb_enu_to_ned(extents) -> list` of six floats. `flat_enu_to_ned(flat: dict) -> dict`, converting `x`, `x_dot`, `x_ddot`, `x_dddot`, `x_ddddot` with `enu_to_ned` and `yaw` / `yaw_dot` with the yaw maps, leaving other keys unchanged. `body_to_sensor_flu_to_frd(q_xyzw) -> (4,)` with `R_new = R_old @ M_FLU_TO_FRD`. `apply_frame(frame, *objects) -> None`, calling `set_frame` only on objects that have it.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.frames import (
    M_ENU_TO_NED, M_FLU_TO_FRD, aabb_enu_to_ned, attitude_enu_flu_to_ned_frd,
    attitude_ned_frd_to_enu_flu, enu_to_ned, flat_enu_to_ned, flu_to_frd,
    frd_to_flu, ned_to_enu, yaw_enu_to_ned, yaw_rate_enu_to_ned,
)

def test_maps_are_involutions_and_yaw_is_not_the_same_number():
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(ned_to_enu(enu_to_ned(v)), v)
    assert np.allclose(enu_to_ned(v), [2.0, 1.0, -3.0])
    assert np.allclose(frd_to_flu(flu_to_frd(v)), v)
    assert np.allclose(flu_to_frd(v), [1.0, -2.0, -3.0])
    assert yaw_enu_to_ned(0.0) == np.pi / 2
    assert yaw_enu_to_ned(np.pi / 2) == 0.0
    assert yaw_rate_enu_to_ned(0.2) == -0.2
    q = np.array([0.0, 0.0, 0.0, 1.0])
    R = Rotation.from_quat(attitude_enu_flu_to_ned_frd(q)).as_matrix()
    assert np.allclose(R, M_ENU_TO_NED @ M_FLU_TO_FRD)
    assert np.allclose(attitude_ned_frd_to_enu_flu(attitude_enu_flu_to_ned_frd(q)), q)

def test_aabb_and_flat_and_homing_map_stays_distinct():
    assert aabb_enu_to_ned([1, 4, -2, 5, -1, 3]) == [-2, 5, 1, 4, -3, 1]
    flat = flat_enu_to_ned({"x": np.array([1.0, 2.0, 3.0]), "yaw": 0.0, "yaw_dot": 0.2, "name": "a"})
    assert np.allclose(flat["x"], [2.0, 1.0, -3.0])
    assert flat["yaw"] == np.pi / 2
    assert flat["name"] == "a"
    from rotorpy.guidance.frames import ned_to_sim
    assert np.allclose(ned_to_sim(np.array([1.0, 2.0, 3.0])), [1.0, 2.0, -3.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frames.py -v`
Expected: FAIL with import error for `rotorpy.frames`

- [ ] **Step 3: Implement the maps in `rotorpy/frames.py`**

`aabb_enu_to_ned` swaps the x interval with the y interval and replaces the z interval `[zmin, zmax]` with `[-zmax, -zmin]`. `apply_frame` ignores `None` and objects without `set_frame`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frames.py -v`
Expected: PASS

---

### Task 2: World stores the frame

**Files:**
- Modify: `rotorpy/world.py` (`World.__init__`, `from_file`, `empty`, `grid_forest`)
- Test: `tests/test_world_frame.py`

**Interfaces:**
- Consumes: `FrameConvention` from Task 1
- Produces: `world.frame`. Constructor argument `frame` defaults to `FrameConvention.FLU_ENU`. `from_file(filename, frame=None)` uses the argument when it is not `None`, otherwise the JSON string `"frame"` when present, otherwise `FLU_ENU`. `empty` and `grid_forest` take the same `frame` argument and pass it through. Extents stored on the world are the numbers the caller passed.

- [ ] **Step 1: Write the failing test**

```python
def test_default_frame_and_file_override(tmp_path):
    world = World({"bounds": {"extents": [0, 1, 0, 1, 0, 1]}, "blocks": []})
    assert world.frame is FrameConvention.FLU_ENU
    path = tmp_path / "w.json"
    path.write_text(json.dumps({"bounds": {"extents": [0, 1, 2, 3, 4, 5]}, "blocks": [], "frame": "frd_ned"}))
    loaded = World.from_file(str(path))
    assert loaded.frame is FrameConvention.FRD_NED
    assert list(loaded.world["bounds"]["extents"]) == [0, 1, 2, 3, 4, 5]
    forced = World.from_file(str(path), frame=FrameConvention.FLU_ENU)
    assert forced.frame is FrameConvention.FLU_ENU
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"bounds": {"extents": [0, 1, 0, 1, 0, 1]}, "blocks": [], "frame": "nwu"}))
    with pytest.raises(ValueError):
        World.from_file(str(bad))
    assert World.empty([0, 1, 0, 1, 0, 1], frame=FrameConvention.FRD_NED).frame is FrameConvention.FRD_NED
    assert World.grid_forest(1, 1, 0.2, 1.0, 1.0, frame=FrameConvention.FRD_NED).frame is FrameConvention.FRD_NED
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_world_frame.py -v`
Expected: FAIL because `frame` is not a constructor argument

- [ ] **Step 3: Store `frame` on `World`**

Parse JSON `"frame"` with `FrameConvention(value)`. An unknown string raises `ValueError`. Do not rewrite extents.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_world_frame.py -v`
Expected: PASS

---

### Task 3: Vehicle geometry and axes

**Files:**
- Modify: `rotorpy/vehicles/multirotor.py` (`Multirotor`)
- Test: `tests/test_frame_vehicle.py`

**Interfaces:**
- Consumes: `FrameConvention`, `M_FLU_TO_FRD` from Task 1
- Produces: `Multirotor.set_frame(frame)`. Attributes `thrust_axis` and `weight`. FLU-ENU: `thrust_axis = [0, 0, 1]`, `weight = [0, 0, -mass*9.81]`. FRD-NED: `thrust_axis = [0, 0, -1]`, `weight = [0, 0, +mass*9.81]`. Active rotor positions are the authored FLU positions times `M_FLU_TO_FRD` under FRD-NED. Inertia is `M @ I_authored @ M` with `M = I` or `M_FLU_TO_FRD`. `c_Dx`, `c_Dy`, `c_Dz` stay at the authored values. Allocation `f_to_TM` / `TM_to_f` is rebuilt from the active positions and `thrust_axis`.

- [ ] **Step 1: Write the failing test**

```python
def test_set_frame_twice_negates_rotor_y_once():
    params = copy.deepcopy(quad_params)
    params["Ixy"] = 0.2
    vehicle = Multirotor(params, aero=False)
    authored_y = quad_params["rotor_pos"]["r1"][1]
    vehicle.set_frame(FrameConvention.FRD_NED)
    vehicle.set_frame(FrameConvention.FRD_NED)
    assert vehicle.rotor_pos["r1"][1] == pytest.approx(-authored_y)
    assert vehicle.inertia[0, 1] == pytest.approx(-0.2)
    assert vehicle.c_Dy == quad_params["c_Dy"]
    assert np.allclose(vehicle.weight, [0, 0, 0.03 * 9.81])
    assert np.allclose(vehicle.thrust_axis, [0, 0, -1])
    vehicle.set_frame(FrameConvention.FLU_ENU)
    assert vehicle.rotor_pos["r1"][1] == pytest.approx(authored_y)
    assert np.allclose(vehicle.weight, [0, 0, -0.03 * 9.81])
```

Use `crazyflie_params.quad_params`. Mass is `0.03`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_vehicle.py::test_set_frame_twice_negates_rotor_y_once -v`
Expected: FAIL because `set_frame` does not exist

- [ ] **Step 3: Implement `set_frame` from a stored FLU copy**

Keep `_rotor_pos_flu` and `_inertia_flu` taken before the first `set_frame`. Rebuild the allocation the same way `__init__` does, with `thrust_axis` in place of `[0, 0, 1]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_vehicle.py::test_set_frame_twice_negates_rotor_y_once -v`
Expected: PASS

---

### Task 4: Hover dynamics and ground contact

**Files:**
- Modify: `rotorpy/vehicles/multirotor.py` (`statedot`, `get_cmd_motor_speeds`, `_on_ground`, ground normal)
- Test: `tests/test_frame_vehicle.py`

**Interfaces:**
- Consumes: `thrust_axis` and `weight` from Task 3
- Produces: FRD-NED hover holds identity attitude and height. `cmd_vel` with zero velocity error uses `F_des = mass * (a_cmd + g_feedforward)` where `g_feedforward` is `[0, 0, +9.81]` in FLU-ENU and `[0, 0, -9.81]` in FRD-NED, and the thrust direction is `R @ thrust_axis`. Desired body +z in FRD-NED is opposite the desired force. Ground: FLU-ENU on ground when `z <= 0.001` and a negative `Fz` is cancelled by `+z`. FRD-NED on ground when `z >= -0.001` and a positive `Fz` is cancelled by `-z`.

- [ ] **Step 1: Write the failing test**

```python
def test_frd_hover_holds_identity_and_height():
    vehicle = Multirotor(quad_params, aero=False)
    vehicle.set_frame(FrameConvention.FRD_NED)
    hover = np.sqrt(0.03 * 9.81 / (4 * quad_params["k_eta"]))
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0.0, 0.0, 0.0, 1.0]),
             "w": np.zeros(3), "wind": np.zeros(3), "rotor_speeds": np.ones(4) * hover}
    for _ in range(100):
        state = vehicle.step(state, {"cmd_motor_speeds": np.ones(4) * hover}, 0.01)
    assert state["x"][2] == pytest.approx(0.0, abs=1e-3)
    assert np.allclose(state["q"], [0, 0, 0, 1], atol=1e-3)

def test_frd_cmd_vel_hover_speeds_match_hover():
    vehicle = Multirotor(quad_params, aero=False, control_abstraction="cmd_vel")
    vehicle.set_frame(FrameConvention.FRD_NED)
    hover = np.sqrt(0.03 * 9.81 / (4 * quad_params["k_eta"]))
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0., 0., 0., 1.]),
             "w": np.zeros(3), "wind": np.zeros(3), "rotor_speeds": np.ones(4) * hover}
    speeds = vehicle.get_cmd_motor_speeds(state, {"cmd_v": np.zeros(3), "cmd_yaw": 0.0})
    assert np.allclose(speeds, np.ones(4) * hover, rtol=1e-2, atol=1.0)

def test_ground_side_follows_down():
    flu = Multirotor(quad_params, aero=False, enable_ground=True)
    frd = Multirotor(quad_params, aero=False, enable_ground=True)
    frd.set_frame(FrameConvention.FRD_NED)
    assert flu._on_ground({"x": np.array([0.0, 0.0, 1.0])}) is False
    assert frd._on_ground({"x": np.array([0.0, 0.0, -1.0])}) is False
    assert frd._on_ground({"x": np.array([0.0, 0.0, 0.01])}) is True
```

If `get_cmd_motor_speeds` for `cmd_vel` does not read `cmd_yaw`, pass the control dict that method already requires and still assert the hover speeds.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_vehicle.py::test_frd_hover_holds_identity_and_height tests/test_frame_vehicle.py::test_ground_side_follows_down -v`
Expected: FAIL because gravity is still minus-z after `set_frame`

- [ ] **Step 3: Use `weight` and `thrust_axis` in `statedot`, `compute_body_wrench`, and `get_cmd_motor_speeds`**

In `compute_body_wrench`, rotor thrust and the yaw reaction both lie along `thrust_axis`, so an FRD hover still pushes up. `cmd_acc` takes a force-like acceleration that already includes gravity compensation. Its thrust projection uses `thrust_axis`, and its desired body +z is `normalize(F_des)` in FLU-ENU and `-normalize(F_des)` in FRD-NED.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_vehicle.py -v`
Expected: PASS

---

### Task 5: SE3 controller

**Files:**
- Modify: `rotorpy/controllers/quadrotor_control.py` (`SE3Control`)
- Test: `tests/test_frame_control.py`

**Interfaces:**
- Consumes: `FrameConvention` from Task 1
- Produces: `SE3Control.set_frame(frame)`. At a level hover (zero error, zero flat acceleration, yaw 0, identity attitude) `cmd_thrust` is about `mass * 9.81` and positive, and `cmd_q` is the identity quaternion, in both frames. FRD-NED adds `[0, 0, -9.81]` per unit mass and points body +z opposite the desired force. Rotor allocation inside the controller is rebuilt from the stored FLU rotor positions, idempotently, the same way as Task 3.

- [ ] **Step 1: Write the failing test**

```python
def test_frd_hover_command_is_identity_and_positive_thrust():
    ctrl = SE3Control(quad_params)
    ctrl.set_frame(FrameConvention.FRD_NED)
    ctrl.set_frame(FrameConvention.FRD_NED)
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0., 0., 0., 1.]), "w": np.zeros(3)}
    flat = {"x": np.zeros(3), "x_dot": np.zeros(3), "x_ddot": np.zeros(3), "yaw": 0.0, "yaw_dot": 0.0}
    cmd = ctrl.update(0.0, state, flat)
    assert cmd["cmd_thrust"] == pytest.approx(0.03 * 9.81, rel=1e-3)
    assert np.allclose(cmd["cmd_q"], [0, 0, 0, 1], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_control.py::test_frd_hover_command_is_identity_and_positive_thrust -v`
Expected: FAIL because `set_frame` does not exist

- [ ] **Step 3: Implement `SE3Control.set_frame`**

Store the FLU rotor positions at init. Gravity feedforward and `b3_des` follow the Task 4 signs.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_control.py -v`
Expected: PASS

---

### Task 6: IMU gravity

**Files:**
- Modify: `rotorpy/sensors/imu.py` (`Imu`, `BatchedImu`)
- Test: `tests/test_frame_imu.py`

**Interfaces:**
- Consumes: `FrameConvention`, `M_FLU_TO_FRD` from Task 1
- Produces: `Imu.set_frame` and `BatchedImu.set_frame`. `gravity_vector` is `[0, 0, -9.81]` or `[0, 0, +9.81]`. The measurement formula stays `R @ (a_world - gravity_vector)`. Under FRD-NED, a noise sample and a bias-walk sample drawn as body vectors are then multiplied by `M_FLU_TO_FRD`. At rest, identity attitude, zero world acceleration, FRD specific force is `[0, 0, -9.81]` when noise is off.

- [ ] **Step 1: Write the failing test**

```python
def test_frd_specific_force_at_rest_is_minus_z():
    imu = Imu(np.zeros(3), np.eye(3), sampling_rate=100)
    imu.set_frame(FrameConvention.FRD_NED)
    state = {"x": np.zeros(3), "v": np.zeros(3), "q": np.array([0., 0., 0., 1.]), "w": np.zeros(3)}
    statedot = {"vdot": np.zeros(3), "wdot": np.zeros(3)}
    meas = imu.measurement(state, statedot, with_noise=False)
    assert np.allclose(meas["accel"], [0, 0, -9.81], atol=1e-6)
```

Add a second test that stubs `np.random.normal` to return `[0.0, 1.0, 2.0]` on every draw and asserts the FRD accel noise contribution is `[0.0, -1.0, -2.0]` relative to the noiseless measurement. A third assertion: `BatchedImu(1, device="cpu").set_frame(FrameConvention.FRD_NED)` leaves `gravity_vector` equal to `[0, 0, 9.81]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_imu.py -v`
Expected: FAIL because `set_frame` does not exist

- [ ] **Step 3: Implement `set_frame` on both IMU classes**

`BatchedImu.gravity_vector` is a tensor of shape `(3,)` on the IMU's device.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_imu.py -v`
Expected: PASS

---

### Task 7: Batched vehicle and controller

**Files:**
- Modify: `rotorpy/vehicles/multirotor.py` (`BatchedMultirotor`)
- Modify: `rotorpy/controllers/quadrotor_control.py` (`BatchedSE3Control`)
- Test: `tests/test_frame_batched.py`

**Interfaces:**
- Consumes: Tasks 3, 4, 5, and 6
- Produces: `BatchedMultirotor.set_frame` and `BatchedSE3Control.set_frame`, with the same signs as the single-vehicle classes. One drone in `cmd_ctatt`, given `cmd_thrust = mass*9.81` and identity `cmd_q`, holds height and identity for 1 s after `set_frame(FRD_NED)`. `set_frame` twice does not double-flip rotor y.

- [ ] **Step 1: Write the failing test**

Use one Crazyflie, `device="cpu"`, `aero=False`, integrator `"rk4"`, `dt=0.01`, 100 steps. Assert final `x[0, 2]` within `1e-2` of 0 and the quaternion within `1e-2` of identity. Assert rotor y of the first rotor after two `set_frame(FRD_NED)` calls equals the negation of the authored y.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_batched.py -v`
Expected: FAIL because batched `set_frame` does not exist

- [ ] **Step 3: Implement batched `set_frame`**

Derive active geometry from a stored FLU copy. `cmd_ctatt` tracking itself does not add gravity. The batched SE3 path does, with the Task 5 signs. Include the batched `cmd_vel` / `cmd_acc` branches if they duplicate the single-vehicle gravity term.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_batched.py -v`
Expected: PASS

---

### Task 8: Wind filter gravity

**Files:**
- Modify: `rotorpy/estimators/wind_ukf.py`
- Modify: `rotorpy/estimators/wind_ekf.py`
- Test: `tests/test_frame_wind_model.py`

**Interfaces:**
- Consumes: `FrameConvention` from Task 1
- Produces: `set_frame(frame)` on `WindUKF` and `WindEKF`. At the zero state (Euler angles 0, body velocity 0, body wind 0) the body-z acceleration of `f` is 0 when the thrust input cancels gravity: FLU-ENU thrust acceleration `+9.81` on the filter's thrust input, FRD-NED thrust acceleration `-9.81`. `g` stays `9.81`. The EKF Jacobian is evaluated at that same hover so its state prediction matches `f`.

- [ ] **Step 1: Write the failing test**

```python
def test_ukf_frd_hover_body_z_accel_is_zero():
    params = copy.deepcopy(quad_params)
    filt = WindUKF(params, dt=0.01)
    filt.set_frame(FrameConvention.FRD_NED)
    x = np.zeros(9)
    filt.uk[0] = -9.81
    x_next = filt.f(x, 0.01)
    assert x_next[5] == pytest.approx(0.0, abs=1e-6)
```

`uk` is the input vector the class already stores. If the thrust component is not index 0, use the index the current FLU hover cancels (`uk[0] = +9.81` leaves `x_next[5] == 0` before this task). Add the same assertion for `WindEKF` at its hover input. The FLU case `uk[0] = +9.81` still yields `x_next[5] == 0`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_wind_model.py -v`
Expected: FAIL because `set_frame` does not exist

- [ ] **Step 3: Select the gravity sign from the frame**

Change the body-z gravity term and the `g` times angle couplings so the FRD hover test passes and the existing FLU hover residual stays zero. Keep the state layout `[psi, theta, phi, body velocity, body wind]`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_wind_model.py tests/test_estimators.py -v`
Expected: PASS

---

### Task 9: Propagate the world frame

**Files:**
- Modify: `rotorpy/environments.py`
- Modify: `rotorpy/simulate.py` (`simulate`, `simulate_batch`)
- Test: `tests/test_frame_propagate.py`

**Interfaces:**
- Consumes: `apply_frame` from Task 1, `World.frame` from Task 2, `set_frame` from Tasks 4, 5, 6, 7, and 8
- Produces: after `Environment(...)`, `simulate(...)`, or `simulate_batch(...)` returns or is constructed, the vehicle, controller, IMU, and estimator that implement `set_frame` are on `world.frame`. A list of worlds passed to `simulate_batch` must all share one frame. Mixed frames raise `ValueError`. Objects without `set_frame` are left alone.

- [ ] **Step 1: Write the failing test**

```python
def test_environment_pushes_frd_onto_vehicle_and_controller():
    world = World.empty([-3, 3, -3, 3, -3, 3], frame=FrameConvention.FRD_NED)
    env = Environment(Multirotor(quad_params, aero=False), SE3Control(quad_params),
                      HoverTraj(), world=world, sim_rate=100)
    assert np.allclose(env.vehicle.weight, [0, 0, 0.03 * 9.81])
    cmd = env.controller.update(0.0, hover_state(), hover_flat())
    assert np.allclose(cmd["cmd_q"], [0, 0, 0, 1], atol=1e-6)
```

`hover_state` and `hover_flat` are the zero-error identity hover from Task 5. Add a `simulate_batch` test with two worlds of different frames and expect `ValueError` before stepping.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_propagate.py -v`
Expected: FAIL because the vehicle weight is still minus-z

- [ ] **Step 3: Call `apply_frame(world.frame, vehicle, controller, imu, estimator)`**

Call it at the end of `Environment.__init__` and at the start of `simulate` and `simulate_batch`. The world's frame replaces a frame set earlier on the object.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_propagate.py -v`
Expected: PASS

---

### Task 10: Multiprocessing worker receives the frame

**Files:**
- Modify: `examples/multimav_multiprocessing_demo.py` (`run_sim`, the pool config)
- Test: `tests/test_frame_propagate.py`

**Interfaces:**
- Consumes: `set_frame` from Tasks 3 and 5
- Produces: `run_sim(trajectory, t_offset, t_final=10, t_step=1/100, frame=FrameConvention.FLU_ENU)`. The worker calls `set_frame` on the `Multirotor` and `SE3Control` it constructs. The script passes `world.frame` from the `World.empty` it already builds. The default argument keeps a caller that omits `frame` on FLU-ENU.

- [ ] **Step 1: Write the failing test**

Call `run_sim(HoverTraj(x0=np.zeros(3)), 0.0, t_final=0.2, t_step=0.01, frame=FrameConvention.FRD_NED)` and assert the last position z is within `1e-2` of the trajectory height and the last quaternion is within `1e-2` of identity. Import `run_sim` from the example module.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_propagate.py::test_run_sim_frd_hover -v`
Expected: FAIL because `run_sim` does not accept `frame`, or the hover climbs

- [ ] **Step 3: Pass `world.frame` into `run_sim` and call `set_frame`**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_propagate.py -v`
Expected: PASS

---

### Task 11: PX4 packing

**Files:**
- Modify: `rotorpy/vehicles/px4_multirotor.py`
- Test: `tests/test_frame_bridges.py`

**Interfaces:**
- Consumes: `FrameConvention`, `attitude_enu_flu_to_ned_frd` from Task 1. Existing `Ardupilot._quaternion_rotorpy_to_aerospace` and `enu_to_geodetic`.
- Produces: `px4_hil_fields(frame, state, accel_body, gyro_body) -> dict` with `lat_deg`, `lon_deg`, `alt_m`, `quaternion` (scalar-first list of 4), `vx_cms`, `vy_cms`, `vz_cms`, `xacc_mg`, `yacc_mg`, `zacc_mg`, `rollspeed`, `pitchspeed`, `yawspeed`, `gyro` (3,). No socket. FLU-ENU matches today's rewrite. FRD-NED skips the axis rewrite. Acceleration is milli-g via `9.80665` in both frames. `rollspeed, pitchspeed, yawspeed` are `state["w"]` in both frames.

- [ ] **Step 1: Write the failing test**

FLU state `x=[10, 20, 5]`, `v=[1, 2, 3]`, identity `q`, `w=[0.1, 0.2, 0.3]`, accel `[0, 0, 9.81]`. Assert `vx_cms, vy_cms, vz_cms` are `200, 100, -300`. Assert `zacc_mg` equals `int(np.round(-9.81 / 9.80665 * 1000))`. Assert angular velocity is `[0.1, 0.2, 0.3]`.

FRD state `x=[20, 10, -5]`, `v=[2, 1, -3]`, identity `q`, same `w`, accel `[0, 0, -9.81]`. Assert the same `200, 100, -300` and the same `zacc_mg`. Assert `quaternion` is `[1, 0, 0, 0]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_bridges.py::test_px4_frd_keeps_millig -v`
Expected: FAIL because `px4_hil_fields` does not exist

- [ ] **Step 3: Implement `px4_hil_fields` and call it from the existing send methods**

FLU quaternion stays `_quaternion_rotorpy_to_aerospace`. FRD quaternion is the inverse of the body-to-world attitude, scalar-first, with no `M_ENU_TO_NED` or `M_FLU_TO_FRD` multiply. Geodetic FRD uses north=`x`, east=`y`, up=`-z`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_bridges.py::test_px4_frd_keeps_millig tests/test_frame_bridges.py::test_px4_flu_matches_current_rewrite -v`
Expected: PASS

---

### Task 12: ArduPilot packing

**Files:**
- Modify: `rotorpy/vehicles/ardupilot_multirotor.py`
- Test: `tests/test_frame_bridges.py`

**Interfaces:**
- Consumes: Task 1 and the existing `_quaternion_rotorpy_to_aerospace`
- Produces: `ardupilot_sensor_fields(frame, state, accel_body, gyro_body) -> dict` with `sim_position`, `sim_velocity_inertial`, `sim_attitude` (scalar-first), `accel` (3,), `gyro` (3,). Position and velocity are `state["x"]` and `state["v"]` in both frames. FLU applies `M_FLU_TO_FRD` to accel and gyro and uses `_quaternion_rotorpy_to_aerospace`. FRD packs accel and gyro as given, and the quaternion is the inverse, scalar-first, with no axis sandwich. Do not construct `Ardupilot` (its initializer opens the plugin).

- [ ] **Step 1: Write the failing test**

Identity FRD state at `x=[4, 5, 6]`, `v=[7, 8, 9]`, accel `[1, 2, 3]`, gyro `[0.1, 0.2, 0.3]`. Assert position and velocity pass through, accel and gyro pass through, and `sim_attitude` is `[1, 0, 0, 0]`. FLU identity with accel `[1, 2, 3]` asserts accel out is `[1, -2, -3]`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_bridges.py::test_ardupilot_frd_skips_axis_rewrite -v`
Expected: FAIL because the helper does not exist

- [ ] **Step 3: Implement the helper and use it from `_create_sensor_data`**

`_create_sensor_data` reads the frame argument it is given, default `FrameConvention.FLU_ENU`, so a current caller that omits it keeps today's fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_bridges.py -v`
Expected: PASS

---

### Task 13: Five-second equivalence flights

**Files:**
- Create: `tests/test_frame_equivalence.py`
- Modify: `pyproject.toml` markers and addopts
- Test: `tests/test_frame_equivalence.py`

**Interfaces:**
- Consumes: Tasks 1 and 9, and Task 10 for the multiprocessing flight
- Produces: pytest marker `frame_equivalence`. Each test runs the scenario in FLU-ENU and in FRD-NED for 5 s, `plot=False`, animation off, IMU noise densities and random walks zero. The FRD run converts initial state, flat outputs, extents, blocks, and wind samples with Task 1. The record is mapped back with `ned_to_enu`, `attitude_ned_frd_to_enu_flu`, and `frd_to_flu`. Final position, velocity, attitude, and body rate meet the global tolerances.

- [ ] **Step 1: Write the failing test**

Three tests:

- `test_basic_usage_matches`: Crazyflie, `SE3Control`, `ThreeDCircularTraj(radius=[3, 3, 0], freq=[0.15, 0.15, 0])`, `SinusoidWind`, world loaded from `rotorpy/worlds/double_pillar.json`, initial state the zero identity state in `examples/basic_usage.py`, `sim_rate=100`, `t_final=5`, `terminate=False`.
- `test_run_sim_matches`: one `TwoDLissajous(A=1, B=1, a=2, b=1, x_offset=-0.5, y_offset=0, height=2.0)` for 5 s through `run_sim`, FLU frame against FRD frame. Wrap the trajectory so FRD flat outputs are `flat_enu_to_ned`.
- `test_batched_matches`: `np.random.seed(10)`, two drones (first Crazyflie, second Hummingbird), `generate_random_minsnap_traj` with 4 waypoints, `v_avg_des=2`, `min_distance=1`, `max_distance=2`, `control_abstraction="cmd_ctatt"`, `BatchedNoWind`, 5 s, `dt=0.01`. Convert each trajectory's waypoints with `enu_to_ned` before building the FRD batch. `motor_noise_std=0`.

Mark all three `@pytest.mark.frame_equivalence`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_equivalence.py -m frame_equivalence -v`
Expected: FAIL because the FRD end state, mapped back, is outside `1e-2`, or the world frame is ignored

- [ ] **Step 3: Add the marker to `pyproject.toml` and fix only integration bugs the tests show**

Do not weaken the tolerances. A trajectory wrapper returns `flat_enu_to_ned` of the original flat output. A wind wrapper returns `enu_to_ned` of the original sample. World extents and blocks for the FRD copy go through `aabb_enu_to_ned`. Initial quaternion goes through `attitude_enu_flu_to_ned_frd`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_equivalence.py -m frame_equivalence -v`
Expected: PASS

---

### Task 14: Camera equivalence

**Files:**
- Modify: `tests/test_frame_equivalence.py`
- Test: `tests/test_frame_equivalence.py`

**Interfaces:**
- Consumes: Tasks 1 and 9. `PinholeCamera.measurement` returns `keypoints` of shape `(N, 2)`.
- Produces: `test_camera_demo_keypoints_match` and `test_camera_visualization_keypoints_match`, both `frame_equivalence`. Keypoints of the FRD render match the FLU render within 2 pixels. The visualization test builds one world with `feature_mode="regular"` and `feature_spacing=0.25` from `WORLD_DATA` in `examples/camera_visualization.py`. It converts extents, blocks, feature points, the parked state, and extrinsics. It does not render the noise grid.

- [ ] **Step 1: Write the failing test**

`test_camera_demo_keypoints_match` uses the example's `double_pillar` world, regular features at spacing `0.08`, the example extrinsics, Crazyflie, `SE3Control`, the example circle, 5 s, and compares `keypoints` of the last camera frame.

`test_camera_visualization_keypoints_match` uses `STATE` and `EXTRINSICS` from that example. FRD position is `enu_to_ned(STATE["x"])`. FRD quaternion is `attitude_enu_flu_to_ned_frd(STATE["q"])`. Extrinsic orientation uses `body_to_sensor_flu_to_frd`. Extrinsic position uses `flu_to_frd`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_frame_equivalence.py::test_camera_visualization_keypoints_match -m frame_equivalence -v`
Expected: FAIL because the FRD keypoints differ by more than 2 pixels

- [ ] **Step 3: Convert the camera inputs with Task 1**

The pinhole class gains no `set_frame`. Feature points on the FRD world are `enu_to_ned` of the FLU points, passed in as embedded features so the generator does not run twice.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_frame_equivalence.py -m frame_equivalence -v`
Expected: PASS

---

### Task 15: Wind-filter trials

**Files:**
- Modify: `examples/wind_filter_example.py`
- Create: `tests/test_frame_wind_trials.py`
- Modify: `pyproject.toml` marker `frame_wind`
- Test: `tests/test_frame_wind_trials.py`

**Interfaces:**
- Consumes: Tasks 8 and 9, `Replay` behavior below
- Produces: thrust removal `accel -= (thrustdes / mass) * thrust_axis` using `vehicle.thrust_axis`, which is `[0, 0, 1]` unless the world frame is FRD-NED. `run_trial_pair(draws) -> tuple` is not required. The test owns the loop. Marker `frame_wind`.

- [ ] **Step 1: Write the failing test**

`test_thrust_removal_follows_axis` in this file, not marked slow: specific force `[0, 0, 9.81]`, thrust `0.03 * 9.81`, mass `0.03`. With `thrust_axis=[0, 0, 1]` the z component becomes `0`. With `thrust_axis=[0, 0, -1]` the z component becomes `19.62`.

`test_fifty_trials_match`, marked `frame_wind`: `np.random.seed(0)`, then draw all 50 trials' scalars and wind means the way `examples/wind_filter_example.py` does, into a list, before either flight. Scalars are copied into the FRD vehicle. Wind mean is `enu_to_ned`. Per-axis intensity swaps x and y and keeps z. For each trial, save `np.random.get_state()`, run the 30 s calibration and 10 s evaluation in FLU-ENU, restore that state, and run FRD-NED. The evaluation wind is a replay of the FLU `state["wind"]` series converted by `enu_to_ned`. IMU `set_frame(FRD_NED)` flips noise and bias-walk samples. Assert per trial the fitted drag within `1e-3`, the three-axis wind RMSE within `1e-2`, the last UKF indexes 6, 7, 8 after `frd_to_flu` within `1e-2`, and the last vehicle state at the global tolerances. `plot=False`. Calibration waypoints are `enu_to_ned` of the script's `points`.

- [ ] **Step 2: Run the fast test to verify it fails**

Run: `pytest tests/test_frame_wind_trials.py::test_thrust_removal_follows_axis -v`
Expected: FAIL because removal is still hardcoded to body z only, or the helper does not exist

- [ ] **Step 3: Remove thrust along `vehicle.thrust_axis` in the example**

Keep the script's default world unset, so a normal run still subtracts from body z. The FRD environment is constructed with `frame=FrameConvention.FRD_NED` inside the test, which makes `thrust_axis` equal `[0, 0, -1]`.

- [ ] **Step 4: Run the fast test, then the fifty-trial test**

Run: `pytest tests/test_frame_wind_trials.py::test_thrust_removal_follows_axis -v`
Expected: PASS

Run: `pytest tests/test_frame_wind_trials.py::test_fifty_trials_match -m frame_wind -v`
Expected: PASS. This runs 50 pairs of 40 s flights. Do not shrink the trial count.

---

### Task 16: Release notes

**Files:**
- Modify: `UPDATES.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the behavior from Tasks 1 through 15
- Produces: `UPDATES.md` version `3.3.0` at the top. `README.md` gains one sentence: a `World` chooses FLU-ENU or FRD-NED, and unset stays FLU-ENU.

- [ ] **Step 1: Write the failing test**

No test. The check is that `UPDATES.md` starts with `## 3.3.0` and names FRD-NED, the world switch, and the example equivalence.

- [ ] **Step 2: Update the two files**

`3.3.0` is a feature bump from `3.2.0`. Do not edit other markdown files.

- [ ] **Step 3: Run the fast suite**

Run: `pytest -q`
Expected: PASS, with `frame_equivalence` and `frame_wind` excluded by addopts
