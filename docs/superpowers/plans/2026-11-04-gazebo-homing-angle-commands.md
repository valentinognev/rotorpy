# Gazebo Homing Angle and Angle-Rate Commands Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the Gazebo homing tracker command PX4 offboard with attitude (angle) or body-rate setpoints, besides today's velocity setpoints. Attitude is the default.

**Architecture:** A new pure module `rotorpy/guidance/offboard_command.py` turns a `HomingCommand` and the measured velocity into an attitude-plus-thrust setpoint, and turns that into body rates. It reuses the PX4 math already ported in `rotorpy/controllers/px4_cascade/math.py` and the tilt-safe acceleration law `homing.track_acceleration`. The law is first proven closed-loop offline through `Px4Cascade`'s `attitude`/`rate` tiers. Then the tracker's 50 Hz setpoint publisher recomputes it from the freshest PX4 state and sends `SET_ATTITUDE_TARGET`. Startup (takeoff, climb, settle) stays on velocity setpoints in every mode.

**Tech Stack:** Python 3.11 (conda env `rotorpy`), numpy, scipy `Rotation`, pymavlink, pytest. Real-Gazebo runs happen in the `px4-noetic-sim-ros` container via `./run_camera_homing_sitl.sh`.

**Spec:** None. The user skipped design questions ("write the plan"). The decisions below are the spec; they are the planner's calls, so reviewers may question them.

## Design decisions (planner's calls)

1. `command_mode` ∈ {`"attitude"` (default), `"rate"`, `"velocity"`}, set by the JSONC key `command_mode`. A missing key means `"attitude"`. Any other value raises `ValueError` at config load.
2. The angle law is PX4's own position-controller back half, fed from the homing command:
   - `acc = track_acceleration(v_meas, v_cmd, pn, tau, max_tilt, min_up)`. `pn = command.acceleration_ned` only when `enable_acceleration_feedforward` is on, else zeros. This keeps the Phase 7 "feedforward off by default" ruling.
   - Then `thrust_from_acceleration`, then `saturate_thrust`, then `thrust_to_attitude(thr, yaw)`.
   - Thrust sent = `-thrust_body_z`, which is normalized and equals hover at zero error.
3. The rate law is `attitude_rate_setpoint(q_current, q_sp, 0, MC_ROLL_P, MC_PITCH_P, MC_YAW_P, MC_YAW_WEIGHT, rate_max)`, with the thrust from (2).
4. Gains come from the live PX4 (`dump_px4_parameters`), read once after connect. Any parameter that does not answer falls back to `px4_cascade.gains.GAINS`, and the fallback is logged.
5. The setpoint is computed in **PX4 estimator NED** (via `EstimatorOriginAlignment`), not in the Gazebo world frame. MAVLink quaternions are scalar-first `[w, x, y, z]`.
6. The setpoint is recomputed on every publisher tick (50 Hz) from the latest `VehicleState`. If attitude or velocity is older than `PROPOSED_TELEMETRY_TIMEOUT_S` or non-finite, that tick **sends nothing**. Offboard loss and the telemetry watchdog then handle it; the tracker never flies a stale angle.
7. Placeholder commands, `StartupCommand`, and every command before `_startup_done` go out as velocity, unchanged.

## Global Constraints

- Test runner: `/home/valento/Anaconda3/envs/rotorpy/bin/python -m pytest`.
- `_send(command)` stays the velocity wire with its current behavior. Existing tests in `tests/test_gazebo_homing_startup.py` must pass unmodified.
- A tracker built via `CameraHomingTracker.__new__` (no `_command_mode` attribute) behaves exactly as today (velocity). Use `getattr(self, "_command_mode", CommandMode.VELOCITY)`. Only the real constructor applies the config default `attitude`.
- Do not change `HomingGuidance`, `SetpointStream`, or `StartupGate` behavior.
- Guidance never reads Gazebo truth (`tests/test_homing_truth_boundary.py` must still pass).
- No `git commit`/`git push` unless the user explicitly authorizes it. "Checkpoint" steps mean: run the task's tests and stage the files.
- At the end: add a top `UPDATES.md` entry `4.15.0`; update `README.md` architecture only where it describes the PX4 setpoint path.

## Review Focus

1. **Frame/order mistakes**: a nonzero `estimator_yaw_offset_deg`, and MAVLink scalar-first quaternions. A wrong yaw offset flies the drone sideways. Test in Task 5: `test_attitude_target_is_in_estimator_frame_with_yaw_offset`.
2. **Stale or missing PX4 state in the publisher thread**: the tracker must send nothing, never a held angle. Test in Task 5: `test_publisher_skips_attitude_target_when_state_is_stale`.
3. **The velocity→attitude handover at `_startup_done`**: at a settled hover, the first attitude target must be level-ish with hover thrust, with no thrust step. Test in Task 5: `test_first_attitude_target_after_startup_matches_hover`.
4. **Missing or wrong PX4 hover thrust**: fall back to `GAINS`. Zero velocity error must give exactly the hover thrust used. Test in Task 2: `test_zero_error_gives_configured_hover_thrust`.
5. **Large attitude error in rate mode**: rates must be clamped to `MC_*RATE_MAX`, and must never be NaN. Test in Task 3: `test_rates_are_clamped_for_large_error`.

---

### Task 1: Command mode selection

**Files:**
- Create: `rotorpy/guidance/offboard_command.py`
- Modify: `rotorpy/guidance/homing_config.py` (add next to `PROPOSED_ENABLE_ACCELERATION_FEEDFORWARD`, ~line 149)
- Modify: `examples/camera_homing.jsonc` (add `"command_mode": "attitude"` with a one-line comment listing the three values)
- Test: `tests/test_offboard_command.py`

**Interfaces:**
- Produces:
  - `class CommandMode(str, Enum)` with members `ATTITUDE="attitude"`, `RATE="rate"`, `VELOCITY="velocity"`.
  - `parse_command_mode(value: str) -> CommandMode`. It raises `ValueError` listing the valid values.
  - `homing_config.PROPOSED_COMMAND_MODE = "attitude"`.
  - `homing_config.command_mode_from_config(config: dict) -> CommandMode`. It uses `config.get("command_mode", PROPOSED_COMMAND_MODE)`.
  - `load_homing_config` calls `command_mode_from_config(config)`, so a bad value fails at load.

- [ ] **Step 1: Write failing tests**

```python
def test_parse_command_mode_accepts_three_modes():
    assert parse_command_mode("attitude") is CommandMode.ATTITUDE
    assert parse_command_mode("rate") is CommandMode.RATE
    assert parse_command_mode("velocity") is CommandMode.VELOCITY

def test_parse_command_mode_rejects_unknown():
    with pytest.raises(ValueError, match="attitude"):
        parse_command_mode("angles")

def test_missing_key_defaults_to_attitude():
    assert command_mode_from_config({}) is CommandMode.ATTITUDE

def test_example_config_selects_attitude():
    assert command_mode_from_config(load_homing_config("examples/camera_homing.jsonc")) is CommandMode.ATTITUDE

def test_load_rejects_bad_mode(tmp_path):
    # copy examples/camera_homing.jsonc, set "command_mode": "bogus"
    with pytest.raises(ValueError):
        load_homing_config(bad_path)
```

- [ ] **Step 2: Run them to verify they fail.** Run `…/python -m pytest tests/test_offboard_command.py -q`. Expected: ImportError/FAIL.
- [ ] **Step 3: Implement** the interfaces above.
- [ ] **Step 4: Run them to verify they pass.** Also run `tests/test_homing_config*.py`. Expected: all pass.
- [ ] **Step 5: Checkpoint.**

---

### Task 2: Attitude (angle) setpoint law

**Files:**
- Modify: `rotorpy/guidance/offboard_command.py`
- Test: `tests/test_offboard_command.py`

**Interfaces:**
- Consumes:
  - `homing.track_acceleration(velocity_sim, goal_sim, pn_sim, tau_s, max_tilt_deg, min_specific_up_g)`
  - `frames.ned_to_sim` / `frames.sim_to_ned`
  - `px4_cascade.math.thrust_from_acceleration`, `saturate_thrust`, `thrust_to_attitude`
  - `px4_cascade.gains.GAINS`
- Produces:
  - `OFFBOARD_PX4_PARAMETERS: tuple[str, ...]` = `("MPC_THR_HOVER","MPC_THR_MIN","MPC_THR_MAX","MPC_THR_XY_MARG","MPC_TILTMAX_AIR","MC_ROLL_P","MC_PITCH_P","MC_YAW_P","MC_YAW_WEIGHT","MC_ROLLRATE_MAX","MC_PITCHRATE_MAX","MC_YAWRATE_MAX")`.
  - `@dataclass(frozen=True) class OffboardGains` with fields:
    - `hover_thrust, thr_min, thr_max, thr_xy_margin, tilt_max_rad`
    - `roll_p, pitch_p, yaw_p, yaw_weight`
    - `rate_limit_rad: np.ndarray (3,)`
    - `track_tau_s, max_tilt_deg, min_specific_up_g`
  - `offboard_gains(config: dict, px4_params: Mapping[str, float]) -> tuple[OffboardGains, list[str]]`:
    - The `list[str]` names the parameters that fell back to `GAINS`.
    - Degrees from PX4 are converted to radians here.
    - `track_tau_s`, `max_tilt_deg`, and `min_specific_up_g` come from `config` keys of the same name.
  - `@dataclass(frozen=True) class AttitudeSetpoint`:
    - `quat_xyzw: np.ndarray`, a scalar-last NED/FRD quaternion in the caller's frame.
    - `thrust: float`, normalized to `[0, 1]`.
    - `body_rates_frd: Optional[np.ndarray] = None`.
  - `attitude_setpoint(velocity_cmd_ned, acceleration_cmd_ned, yaw_rad, measured_velocity_ned, gains: OffboardGains, enable_acceleration_feedforward: bool = False) -> AttitudeSetpoint`:
    - It is frame-agnostic: all vectors are in one NED frame, and the yaw is in that frame.
    - It raises `ValueError` on any non-finite input.
    - The algorithm is decision 2, verbatim.

- [ ] **Step 1: Write failing tests**

```python
def _gains(**kw): # GAINS defaults + config tau 0.5, tilt 45, min_up 0.3
    ...

def test_offboard_gains_prefer_px4_and_report_fallbacks():
    g, fallback = offboard_gains(CONFIG, {"MPC_THR_HOVER": 0.62, "MC_ROLLRATE_MAX": 180.0})
    assert g.hover_thrust == 0.62
    assert g.rate_limit_rad[0] == pytest.approx(np.deg2rad(180.0))
    assert g.roll_p == GAINS["MC_ROLL_P"] and "MC_ROLL_P" in fallback and "MPC_THR_HOVER" not in fallback

def test_zero_error_gives_configured_hover_thrust():          # Review Focus 4
    sp = attitude_setpoint([0,0,0], [0,0,0], 0.3, [0,0,0], _gains(hover_thrust=0.6))
    assert sp.thrust == pytest.approx(0.6)
    yaw, pitch, roll = Rotation.from_quat(sp.quat_xyzw).as_euler("ZYX")
    assert (yaw, pitch, roll) == pytest.approx((0.3, 0.0, 0.0), abs=1e-9)

def test_north_velocity_demand_pitches_nose_down():
    sp = attitude_setpoint([5,0,0], [0,0,0], 0.0, [0,0,0], _gains())
    _, pitch, roll = Rotation.from_quat(sp.quat_xyzw).as_euler("ZYX")
    assert pitch < 0 and abs(roll) < 1e-9 and sp.thrust > 0.5

def test_climb_demand_raises_thrust_and_descent_lowers_it():
    assert attitude_setpoint([0,0,-3],[0,0,0],0,[0,0,0],_gains()).thrust > 0.5
    assert attitude_setpoint([0,0, 3],[0,0,0],0,[0,0,0],_gains()).thrust < 0.5

def test_tilt_never_exceeds_max_tilt():
    sp = attitude_setpoint([100,100,0],[0,0,0],0,[0,0,0],_gains())
    body_z = Rotation.from_quat(sp.quat_xyzw).apply([0,0,1])
    assert np.degrees(np.arccos(body_z[2])) <= 45.0 + 1e-6

def test_feedforward_ignored_unless_enabled():
    a = attitude_setpoint([0,0,0],[3,0,0],0,[0,0,0],_gains())
    b = attitude_setpoint([0,0,0],[3,0,0],0,[0,0,0],_gains(), enable_acceleration_feedforward=True)
    assert np.allclose(a.quat_xyzw, attitude_setpoint([0,0,0],[0,0,0],0,[0,0,0],_gains()).quat_xyzw)
    assert not np.allclose(a.quat_xyzw, b.quat_xyzw)

def test_non_finite_input_raises():
    with pytest.raises(ValueError):
        attitude_setpoint([np.nan,0,0],[0,0,0],0,[0,0,0],_gains())
```

- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement** `OffboardGains`, `offboard_gains`, `AttitudeSetpoint`, and `attitude_setpoint`. If `thrust_to_attitude`'s quaternion sign flips, canonicalize it to `w >= 0` so consecutive setpoints are continuous.
- [ ] **Step 4: Run them to verify they pass.** Also run `tests/test_px4_math.py`; it must stay green.
- [ ] **Step 5: Checkpoint.**

---

### Task 3: Body-rate (angle-rate) setpoint law

**Files:**
- Modify: `rotorpy/guidance/offboard_command.py`
- Test: `tests/test_offboard_command.py`

**Interfaces:**
- Consumes: `px4_cascade.math.attitude_rate_setpoint`, plus `AttitudeSetpoint` and `OffboardGains` from Task 2.
- Produces: `rate_setpoint(setpoint: AttitudeSetpoint, current_quat_xyzw, gains: OffboardGains) -> AttitudeSetpoint`.
  - It returns a copy with `body_rates_frd` filled (FRD rad/s) and the thrust unchanged.
  - Both quaternions must be in the same NED frame.
  - It raises `ValueError` on a non-finite or zero-norm `current_quat_xyzw`.

- [ ] **Step 1: Write failing tests**

```python
def test_rates_zero_when_at_setpoint():
    sp = attitude_setpoint([0,0,0],[0,0,0],0.4,[0,0,0],_gains())
    assert np.allclose(rate_setpoint(sp, sp.quat_xyzw, _gains()).body_rates_frd, 0.0, atol=1e-9)

def test_rates_point_toward_setpoint():
    sp = attitude_setpoint([5,0,0],[0,0,0],0.0,[0,0,0],_gains())   # nose-down
    rates = rate_setpoint(sp, [0,0,0,1], _gains()).body_rates_frd
    assert rates[1] < 0 and rate_setpoint(sp, [0,0,0,1], _gains()).thrust == sp.thrust

def test_rates_are_clamped_for_large_error():                 # Review Focus 5
    sp = attitude_setpoint([0,0,0],[0,0,0],np.pi,[0,0,0],_gains())
    current = Rotation.from_euler("ZYX",[0,0,np.radians(80)]).as_quat()
    r = rate_setpoint(sp, current, _gains()).body_rates_frd
    assert np.all(np.isfinite(r)) and np.all(np.abs(r) <= _gains().rate_limit_rad + 1e-9)

def test_bad_current_attitude_raises():
    with pytest.raises(ValueError):
        rate_setpoint(attitude_setpoint([0,0,0],[0,0,0],0,[0,0,0],_gains()), [0,0,0,0], _gains())
```

- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement** `rate_setpoint`. Pass `yaw_rate=0.0`; the limits come from `gains.rate_limit_rad`.
- [ ] **Step 4: Run them to verify they pass.**
- [ ] **Step 5: Checkpoint.**

---

### Task 4: Offline closed-loop proof through `Px4Cascade` attitude/rate tiers

This task proves the law flies before Gazebo is touched. The offline `pid` controller is `Px4Cascade`, which already has `attitude` and `rate` tiers.

**Files:**
- Modify: `rotorpy/guidance/runner.py` (`control_from_homing`, `px4_command_from_homing`)
- Test: `tests/test_homing_runner.py`, `tests/test_homing_e2e.py`

**Interfaces:**
- Consumes: Tasks 1–3. Also `Px4Cascade.update(t, state, {"tier": "attitude", "attitude", "thrust"})` and `{"tier": "rate", "rate_frd", "thrust"}`.
- Produces:
  - `px4_command_from_homing(command, mode: CommandMode = CommandMode.VELOCITY, measured_velocity_cascade_ned=None, state_q_cascade_ned=None, gains: Optional[OffboardGains] = None) -> dict`.
    - With `VELOCITY` its output is unchanged.
    - With the other modes it feeds `attitude_setpoint` with **exactly the cascade-frame values the velocity tier gets today**: `velocity_ned`/`acceleration_ned` swapped `[1],[0],[2]`, `yaw = π/2 − command.yaw_rad`, and measured `enu_to_ned(state["v"])`.
    - Rate mode passes `attitude_enu_flu_to_ned_frd(state["q"])` to `rate_setpoint`.
  - `control_from_homing` reads `command_mode_from_config(config)` when `kind == "pid"` and `config` is given, else `VELOCITY`.
  - Its gains are `offboard_gains(config, {})[0]` with `track_tau_s`/`max_tilt_deg`/`min_specific_up_g` from config. If `config` is None, use the module defaults `0.5/45/0.3`.

- [ ] **Step 1: Write failing tests** (`tests/test_homing_runner.py`)

```python
def test_attitude_command_matches_px4_ned_at_the_no_lock_start():
    # same setup as test_homing_command_matches_px4_ned_at_the_no_lock_start
    px4 = px4_command_from_homing(command, CommandMode.ATTITUDE,
            measured_velocity_cascade_ned=enu_to_ned(state["v"]), gains=default_gains)
    assert px4["tier"] == "attitude"
    ned_yaw, pitch, roll = Rotation.from_quat(px4["attitude"]).as_euler("ZYX")
    state_yaw = Rotation.from_quat(attitude_enu_flu_to_ned_frd(state["q"])).as_euler("ZYX")[0]
    assert ned_yaw == pytest.approx(state_yaw) and abs(pitch) < 1e-6 and abs(roll) < 1e-6
    assert px4["thrust"] == pytest.approx(default_gains.hover_thrust)

def test_rate_command_is_zero_at_the_no_lock_start():
    px4 = px4_command_from_homing(command, CommandMode.RATE, ..., state_q_cascade_ned=attitude_enu_flu_to_ned_frd(state["q"]))
    assert px4["tier"] == "rate" and np.allclose(px4["rate_frd"], 0.0, atol=1e-6)
```

`tests/test_homing_e2e.py` (mark `@pytest.mark.homing_e2e`):

```python
@pytest.mark.parametrize("mode", ["attitude", "rate"])
def test_angle_modes_home_as_well_as_velocity(mode):
    base = load_homing_config("examples/camera_homing.jsonc") | {"controller": "pid"}
    velocity = run_homing(config=base | {"command_mode": "velocity"})
    angle = run_homing(config=base | {"command_mode": mode})
    assert not angle.timed_out
    assert angle.miss_m <= max(2.0, (velocity.miss_m or 0.0) + 1.0)
```

- [ ] **Step 2: Run them to verify they fail.** Run `…/python -m pytest tests/test_homing_runner.py -q`, then `…/python -m pytest tests/test_homing_e2e.py -q -m homing_e2e`. Pass `-m` explicitly, because `pyproject.toml` `addopts` deselects `homing_e2e` by default.
- [ ] **Step 3: Record the velocity baseline first.** Run only the velocity half and note its `miss_m`/`timed_out` in the task report. If velocity itself times out, stop and report (BLOCKED), because the comparison bound is then meaningless.
- [ ] **Step 4: Implement** the runner changes.
- [ ] **Step 5: Run the tests to verify they pass.** Also run the full `tests/test_homing_runner.py tests/test_px4_cascade.py`. If the closed loop misses, tune nothing silently. Report the miss and the mode; candidates are `track_tau_s` and hover thrust.
- [ ] **Step 6: Checkpoint.**

---

### Task 5: Tracker wiring — `SET_ATTITUDE_TARGET` from the publisher thread

**Files:**
- Modify: `Dockerfiles/camera_homing_tracker.py`:
  - `__init__` (~line 205)
  - `_connect` (~line 294)
  - `_publish_setpoint` / new `_send_attitude_target` (~line 886)
  - `_MAVLINK_POLL_TYPES` unchanged
- Modify: `rotorpy/guidance/vehicle_state.py` (add the read-only `alignment` property returning the `EstimatorOriginAlignment`)
- Test: `tests/test_gazebo_homing_angle_commands.py` (new; reuse the `tracker_module`/`_new_tracker`/`_ingest_estimate` helpers from `tests/test_gazebo_homing_startup.py` by importing them, or move them to a shared `tests/conftest.py` fixture if import is awkward)

**Interfaces:**
- Consumes: Tasks 1–3, `VehicleState.sample_at(None, now_wall_s)`, `EstimatorOriginAlignment.world_vector_to_estimator`, `.heading_to_estimator`, `.attitude_to_estimator`, and `px4_characterization.dump_px4_parameters(connection, names)`.
- Produces (tracker attributes and methods):
  - `self._command_mode: CommandMode`, from `command_mode_from_config(config)`.
  - `self._offboard_gains: OffboardGains`, set in `_connect` after the heartbeat. Each fallback name is logged once via `rospy.logwarn`.
  - `_attitude_target_for(command, now_wall_s) -> Optional[AttitudeSetpoint]`:
    - It returns the estimator-frame setpoint, with rates filled in rate mode.
    - It returns `None` if the sample is missing, older than `PROPOSED_TELEMETRY_TIMEOUT_S`, or non-finite (catches the `ValueError`).
  - `_send_attitude_target(setpoint: AttitudeSetpoint) -> None`:
    - It calls `self._mav.mav.set_attitude_target_send(0, sys, comp, type_mask, [w,x,y,z], r, p, y, thrust)` under `_mav_lock`.
    - Attitude mode uses `type_mask = ROLL|PITCH|YAW_RATE_IGNORE`.
    - Rate mode uses `ATTITUDE_IGNORE`.
    - It stores `self._last_transmitted_attitude = {"mode", "quat_xyzw", "thrust", "body_rates_frd"}` under `_flags_lock`, and sets `self._last_transmitted = None`.
  - `_publish_setpoint(command)` decides the route:
    - Velocity, through the unchanged `_send`, if the mode is `VELOCITY`, or the command is not a `HomingCommand`, or `_startup_done` is false.
    - Otherwise it calls `_attitude_target_for`. A `None` result increments `self._attitude_skips` and sends nothing.

- [ ] **Step 1: Write failing tests**

```python
def test_default_constructed_tracker_uses_attitude_mode(...):  # config without key
def test_bare_tracker_without_mode_still_sends_velocity(...):  # __new__ tracker -> _send called, no attitude send

def test_startup_command_goes_out_as_velocity_in_attitude_mode(...):
    # _command_mode=ATTITUDE, _startup_done=False, StartupCommand -> set_position_target_local_ned_send only

def test_first_attitude_target_after_startup_matches_hover(...):     # Review Focus 3
    # level attitude, zero velocity, HomingCommand(velocity 0, yaw = current yaw), _startup_done=True
    # -> one set_attitude_target_send; mask == 7; thrust == gains.hover_thrust ±0.02;
    #    q (w-first) ≈ current estimator attitude (angle diff < 1°)

def test_attitude_target_is_in_estimator_frame_with_yaw_offset(...): # Review Focus 1
    # alignment yaw_offset 90°; command yaw = world 0 -> sent quaternion yaw == -90° (estimator)
    # command velocity north (world) -> sent tilt direction is along estimator -east axis accordingly

def test_rate_mode_sends_body_rates_with_attitude_ignored(...):
    # mask == ATTITUDE_TARGET_TYPEMASK_ATTITUDE_IGNORE; rates finite; nose-down demand -> pitch rate < 0

def test_publisher_skips_attitude_target_when_state_is_stale(...):   # Review Focus 2
    # attitude older than PROPOSED_TELEMETRY_TIMEOUT_S -> no send of either kind; _attitude_skips == 1

def test_offboard_gains_read_from_px4_with_fallback(...):
    # fake mav answering only MPC_THR_HOVER=0.42 -> _offboard_gains.hover_thrust == 0.42, others == GAINS

def test_attitude_send_is_serialized_with_mav_lock(...):           # mirror the existing _send lock test
```

- [ ] **Step 2: Run them to verify they fail.** Run `…/python -m pytest tests/test_gazebo_homing_angle_commands.py -q`.
- [ ] **Step 3: Implement** per the Interfaces. Convert world→estimator as follows:
  - `v_cmd`, `a_cmd`, `v_meas` via `world_vector_to_estimator`.
  - `yaw` via `heading_to_estimator`.
  - The current attitude via `attitude_to_estimator(sample.attitude_quat_world)`.
- [ ] **Step 4: Run the tests to verify they pass.** Then run the regression suite: `tests/test_gazebo_homing_startup.py tests/test_homing_truth_boundary.py tests/test_setpoint_stream.py tests/test_homing_startup_gate.py -q`. All must pass unmodified.
- [ ] **Step 5: Checkpoint.**

---

### Task 6: Telemetry and replay of transmitted angle commands

**Files:**
- Modify: `rotorpy/guidance/telemetry.py` (`TickRecord`, ~line 191)
- Modify: `Dockerfiles/camera_homing_tracker.py` (`_record_step`, ~line 835)
- Test: `tests/test_homing_telemetry.py`

**Interfaces:**
- Produces new `TickRecord` fields, all defaulting to `None`:
  - `command_mode: Optional[str]`
  - `transmitted_attitude_quat: Optional[List[float]]` (scalar-last, estimator NED/FRD)
  - `transmitted_thrust: Optional[float]`
  - `transmitted_body_rates_frd: Optional[List[float]]`
- `_record_step` fills them from `_last_transmitted_attitude`. `transmitted_setpoint` stays `None` on attitude/rate ticks.

- [ ] **Step 1: Write failing tests**
  - A `TickRecord` holding the four fields round-trips through `TelemetryLog` → `read_telemetry` unchanged.
  - A JSONL line written **without** these keys (pre-4.15 log) still loads, with the fields `None`.
  - A `_record_step` with a stored `_last_transmitted_attitude` emits them, and its `transmitted_setpoint is None`.
- [ ] **Step 2: Run them to verify they fail.**
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run them to verify they pass.** Also run `tests/test_homing_replay.py tests/test_homing_recorded_handoff_replay.py`.
- [ ] **Step 5: Checkpoint.**

---

### Task 7: Real-Gazebo verification and documentation

**Files:**
- Modify: `UPDATES.md` (new top entry `## 4.15.0 - Attitude and body-rate offboard commands for Gazebo homing`)
- Modify: `README.md` (only the paragraph(s) describing the tracker's PX4 velocity setpoints)

- [ ] **Step 1: Full suite.** Run `…/python -m pytest -q` and compare it against the 4.14.0 baseline: the same pre-existing failures are allowed, and there must be 0 new failures. Use `git stash` to compare if counts differ.
- [ ] **Step 2: Real Gazebo, each mode.** Run `HEADLESS=1 ./run_camera_homing_sitl.sh` with `command_mode` set to `attitude`, `rate`, and `velocity`, in turn. For each run, record:
  - the outcome status;
  - the miss;
  - whether `offboard_lost` occurred;
  - `_attitude_skips`;
  - the max tilt and thrust from telemetry around the startup→homing handover.

  Pass criteria:
  - No crash or flip at the handover.
  - No `offboard_lost` caused by the new path.
  - Results are reported honestly, even when a mode homes worse than `velocity`.
- [ ] **Step 3: Docs.** The `UPDATES.md` entry should, in a few laconic lines:
  - name the three modes and the default;
  - describe the law (decisions 2–3) and the estimator frame;
  - explain the stale-state skip and that startup stays velocity;
  - give the offline and Gazebo results.

  `README.md`: state that the tracker sends attitude, rate, or velocity offboard setpoints, selected by `command_mode`.
- [ ] **Step 4: Checkpoint.** Ask the user before any commit.

---

## Reviewer gates (per task)

For each task, the reviewer checks:

- **Spec review:** Does the code match this task's Interfaces exactly, in names, signatures, frames, and quaternion order? Does each Review Focus test owned by this task exist and assert what is listed?
- **Quality review:**
  - No duplicated PX4 math: reuse `px4_cascade.math`.
  - No Gazebo truth on the control path.
  - No behavior change for `__new__`-built trackers or velocity mode.
  - Thread safety: `_mav_lock` for MAVLink, `_flags_lock` for shared flags.
- **Final whole-branch review:** Re-run Task 7 Step 1. Read the handover path end-to-end, from `step()` → `SetpointStream` → `_publish_setpoint` → MAVLink.
