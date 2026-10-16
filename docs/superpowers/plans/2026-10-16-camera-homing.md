# Camera Homing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Hummingbird example that homes on a red box with a fixed camera and reports miss distance at ground contact, plus a navigation-constant sweep that prints how that constant moves the miss.

**Architecture:** Pure guidance functions live under `rotorpy/guidance/` and are tested without a flight. A custom runner steps `Multirotor` and `SE3Control` itself so the target box is not a collision and 5 megapixel frames are discarded after the pixel count and centroid are read. The example and the end-to-end test call that runner.

**Tech Stack:** Python >= 3.8, NumPy, SciPy `Rotation`, existing RotorPy `Multirotor`, `SE3Control`, `PinholeCamera`, `World`, pytest.

**Spec:** `docs/superpowers/specs/2026-10-16-camera-homing-design.md`

## Global Constraints

- Python >= 3.8. Type hints use `typing.Optional` and `typing.Tuple`, not the `|` union syntax.
- Do not modify `rotorpy/vehicles/hummingbird_params.py` or the over-speed check in `rotorpy/simulate.py`.
- Do not call `simulate` or `Environment.run`. Both collision-check world blocks and keep every camera frame.
- Nominal parameters: Hummingbird; target ground NED (100, 50, 0); start NED (0, 0, −30); initial speed 10 m/s horizontal toward that ground point; box 0.5 m by 0.5 m by 1 m, color `[1, 0, 0]`; camera at the body origin, pitched 10° down in body FRD; FOV 90° by 90°; image 2236 by 2236; lock at 36 target pixels; no-lock limit 20 s; locked limit 120 s; navigation constant 3; sweep values 2, 3, 4, 5.
- Miss is the distance from the drone to NED (100, 50, −0.5) at the first ground sample. A timeout has no miss.
- Simulator world is z-up. NED (n, e, d) maps to simulator (n, e, −d). Body FRD (forward, right, down) maps to simulator body (forward, left, up) by negating y and z.
- Maximum command speed: every NED component at most 19 m/s in absolute value. Over-speed timeout: any simulator velocity component strictly greater than 20 m/s.
- Sim rate 100 Hz. Camera and homing update 20 Hz. Images are float arrays in `[0, 1]`, shape `(H, W, 3)`, and are not stored.
- No wind. Height, attitude, and velocity passed into the guidance are the vehicle state.
- Do not git commit or git push. The user has not asked for commits.
- Before Wave 1, the parent creates an empty `rotorpy/guidance/__init__.py` if it is missing. Wave 1 tasks must not edit that file.

## File structure

- `rotorpy/guidance/frames.py` — NED and FRD conversions.
- `rotorpy/guidance/pixels.py` — target mask, count, centroid.
- `rotorpy/guidance/sightline.py` — ground-plane relative position.
- `rotorpy/guidance/navigation.py` — proportional-navigation acceleration.
- `rotorpy/guidance/speed.py` — component speed cap.
- `rotorpy/guidance/outcome.py` — miss distance and timeout.
- `rotorpy/guidance/camera_geometry.py` — resolution, intrinsics, 10° extrinsics, pixel ray.
- `rotorpy/guidance/homing.py` — lock, coast, loss, yaw, and the SE3 flat output.
- `rotorpy/guidance/scene.py` — nominal world, camera, and initial state.
- `rotorpy/guidance/runner.py` — stop rules, one flight, sweep, print lines.
- `examples/camera_homing.py` — nominal flight, `--sweep` for the four constants.
- `tests/test_homing_*.py` — one file per wave-1 module, plus guidance, scene, runner, and the slow end-to-end file.
- `pyproject.toml` — `homing_e2e` marker excluded from the default suite.
- `UPDATES.md` — version 3.1.0. `README.md` — one sentence naming the example.

## Parallel execution

Wave 1 (together): Tasks 1–7. No task imports another guidance module.

Wave 2 (together, after wave 1): Tasks 8 and 9.

Wave 3: Task 10, after wave 2.

Wave 4: Task 11, after wave 3.

A reviewer checks a task before the next wave uses it. Reviewers for tasks in the same wave can run together.

## Review Focus

- A lost track stays lost. A later frame that sees the box again does not resume homing. Tested in Task 8.
- The line-of-sight rate uses the time between camera frames. A sim-step call with no new image does not change the stored sightline. Tested in Task 8.
- Thirty-six target pixels lock the flight even when the sightline misses the ground. That case holds the initial coast velocity and zero acceleration, and the runner’s locked clock is the 120 s clock. Tested in Tasks 8 and 10.
- Ground contact outranks over-speed. A locked sample with simulator z <= 0 is a miss even if a velocity component exceeds 20 m/s. Tested in Task 10.
- The 45° speed command puts 19 m/s on north and on down, magnitude 19√2. Tested in Task 5.

---

### Task 1: Frame conversions

**Files:**
- Create: `rotorpy/guidance/frames.py`
- Test: `tests/test_homing_frames.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ned_to_sim(ned: np.ndarray) -> np.ndarray`, `sim_to_ned(sim: np.ndarray) -> np.ndarray`, `frd_to_sim_body(frd: np.ndarray) -> np.ndarray`, `sim_body_to_frd(body: np.ndarray) -> np.ndarray`. Each takes and returns shape `(3,)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.frames import frd_to_sim_body, ned_to_sim, sim_body_to_frd, sim_to_ned

def test_start_point_ned_to_sim():
    sim = ned_to_sim(np.array([0.0, 0.0, -30.0]))
    assert np.allclose(sim, [0.0, 0.0, 30.0])
    assert np.allclose(sim_to_ned(sim), [0.0, 0.0, -30.0])

def test_frd_flips_right_and_down():
    body = frd_to_sim_body(np.array([1.0, 2.0, 3.0]))
    assert np.allclose(body, [1.0, -2.0, -3.0])
    assert np.allclose(sim_body_to_frd(body), [1.0, 2.0, 3.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_frames.py -v`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `rotorpy.guidance.frames`.

- [ ] **Step 3: Implement the four functions in `rotorpy/guidance/frames.py`**

NED z is negated. FRD y and z are negated. Copy the input array; do not modify it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_frames.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 2: Target pixels

**Files:**
- Create: `rotorpy/guidance/pixels.py`
- Test: `tests/test_homing_pixels.py`

**Interfaces:**
- Consumes: nothing
- Produces: `target_pixel_count(image: np.ndarray) -> int`, `target_centroid(image: np.ndarray) -> Optional[np.ndarray]`. Centroid is shape `(2,)` as `(u, v)` with `u` the column and `v` the row, or `None` when the count is 0. A target pixel has red > 0.5 and red strictly greater than green and blue. `image` is float `(H, W, 3)` in `[0, 1]`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.pixels import target_centroid, target_pixel_count

def test_empty_image_has_no_target():
    image = np.zeros((4, 4, 3))
    assert target_pixel_count(image) == 0
    assert target_centroid(image) is None

def test_red_threshold_and_centroid():
    image = np.zeros((4, 5, 3))
    image[1, 2] = [0.51, 0.5, 0.5]
    image[1, 3] = [0.5, 0.0, 0.0]       # red == 0.5 does not count
    image[2, 2] = [0.9, 0.9, 0.0]       # red is not strictly greater than green
    assert target_pixel_count(image) == 1
    assert np.allclose(target_centroid(image), [2.0, 1.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_pixels.py -v`

Expected: FAIL with import error for `rotorpy.guidance.pixels`.

- [ ] **Step 3: Implement `target_pixel_count` and `target_centroid` in `rotorpy/guidance/pixels.py`**

Centroid is the mean of the matching `(column, row)` coordinates.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_pixels.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 3: Ground sightline

**Files:**
- Create: `rotorpy/guidance/sightline.py`
- Test: `tests/test_homing_sightline.py`

**Interfaces:**
- Consumes: nothing
- Produces: `ground_relative_position(height_m: float, sightline_ned: np.ndarray) -> Optional[np.ndarray]`. Sightline is a NED direction, not necessarily a unit vector. Positive NED z points down. When `height_m > 0` and the sightline’s down-component is positive, return `s * unit_sightline` with `s = height_m / unit_down`. Otherwise return `None`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.sightline import ground_relative_position

def test_straight_down_range_is_the_height():
    relative = ground_relative_position(30.0, np.array([0.0, 0.0, 2.0]))
    assert np.allclose(relative, [0.0, 0.0, 30.0])

def test_scaled_sightline():
    relative = ground_relative_position(30.0, np.array([3.0, 0.0, 4.0]))
    assert np.allclose(relative, [22.5, 0.0, 30.0])

def test_level_or_upward_sightline_is_lost():
    assert ground_relative_position(30.0, np.array([1.0, 0.0, 0.0])) is None
    assert ground_relative_position(30.0, np.array([1.0, 0.0, -0.1])) is None
    assert ground_relative_position(0.0, np.array([0.0, 0.0, 1.0])) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_sightline.py -v`

Expected: FAIL with import error for `rotorpy.guidance.sightline`.

- [ ] **Step 3: Implement `ground_relative_position` in `rotorpy/guidance/sightline.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_sightline.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 4: Proportional navigation

**Files:**
- Create: `rotorpy/guidance/navigation.py`
- Test: `tests/test_homing_navigation.py`

**Interfaces:**
- Consumes: nothing
- Produces: `pn_acceleration(nav_constant: float, velocity_ned: np.ndarray, sightline_ned: np.ndarray, previous_sightline_ned: Optional[np.ndarray], dt: float) -> np.ndarray` of shape `(3,)`.

The sightline arguments are directions from the drone toward the target. Let `u` and `u_prev` be their unit vectors. Closing speed is `dot(u, velocity_ned)` (positive when flying toward the target). When `previous_sightline_ned` is `None` or `dt <= 0`, return zeros. Otherwise `u_dot = (u - u_prev) / dt`, `omega = cross(u, u_dot)`, and acceleration is `nav_constant * closing_speed * cross(omega, u)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.navigation import pn_acceleration

def test_first_sample_and_zero_dt_are_zero():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([4.0, 0.0, 0.0])
    assert np.allclose(pn_acceleration(3.0, v, u, None, 0.5), 0.0)
    assert np.allclose(pn_acceleration(3.0, v, u, np.array([0.0, 1.0, 0.0]), 0.0), 0.0)

def test_known_acceleration_scales_with_n_and_closing_speed():
    u = np.array([1.0, 0.0, 0.0])
    u_prev = np.array([0.0, 1.0, 0.0])
    v = np.array([4.0, 0.0, 0.0])
    accel = pn_acceleration(3.0, v, u, u_prev, 0.5)
    assert np.allclose(accel, [0.0, 24.0, 0.0])
    assert np.allclose(pn_acceleration(6.0, v, u, u_prev, 0.5), [0.0, 48.0, 0.0])
    assert np.allclose(pn_acceleration(3.0, [8.0, 0.0, 0.0], u, u_prev, 0.5), [0.0, 48.0, 0.0])
    assert np.dot(accel, u) == 0.0

def test_zero_closing_speed_is_zero_acceleration():
    accel = pn_acceleration(3.0, np.array([0.0, 5.0, 0.0]), np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 0.5)
    assert np.allclose(accel, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_navigation.py -v`

Expected: FAIL with import error for `rotorpy.guidance.navigation`.

- [ ] **Step 3: Implement `pn_acceleration` in `rotorpy/guidance/navigation.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_navigation.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 5: Speed cap

**Files:**
- Create: `rotorpy/guidance/speed.py`
- Test: `tests/test_homing_speed.py`

**Interfaces:**
- Consumes: nothing
- Produces: `COMPONENT_CAP = 19.0` and `desired_velocity(direction_ned: np.ndarray) -> np.ndarray` of shape `(3,)`. The direction is not necessarily a unit vector. The result is parallel to it, and it is the longest such vector whose components are each at most 19 m/s in absolute value. A zero direction returns zeros.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.speed import desired_velocity

def test_axis_aligned_speed_is_19():
    assert np.allclose(desired_velocity(np.array([2.0, 0.0, 0.0])), [19.0, 0.0, 0.0])
    assert np.allclose(desired_velocity(np.array([0.0, 0.0, -5.0])), [0.0, 0.0, -19.0])

def test_equal_north_and_down_components_are_each_19():
    velocity = desired_velocity(np.array([1.0, 0.0, 1.0]))
    assert np.allclose(velocity, [19.0, 0.0, 19.0])
    assert np.isclose(np.linalg.norm(velocity), 19.0 * np.sqrt(2.0))

def test_zero_direction():
    assert np.allclose(desired_velocity(np.zeros(3)), 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_speed.py -v`

Expected: FAIL with import error for `rotorpy.guidance.speed`.

- [ ] **Step 3: Implement `desired_velocity` in `rotorpy/guidance/speed.py`**

Scale the unit direction by `COMPONENT_CAP / max(abs(unit_components))`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_speed.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 6: Miss and timeout

**Files:**
- Create: `rotorpy/guidance/outcome.py`
- Test: `tests/test_homing_outcome.py`

**Interfaces:**
- Consumes: nothing
- Produces: `BOX_CENTER_NED = np.array([100.0, 50.0, -0.5])`. `HomingOutcome` has `miss_m: Optional[float]`, `timed_out: bool`, `time_s: float`. `miss_outcome(position_ned: np.ndarray, time_s: float) -> HomingOutcome` sets `miss_m` to the Euclidean distance from `position_ned` to `BOX_CENTER_NED` and `timed_out` False. `timeout_outcome(time_s: float) -> HomingOutcome` sets `miss_m` to `None` and `timed_out` True.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.outcome import miss_outcome, timeout_outcome

def test_ground_point_is_half_a_meter_from_the_volume_center():
    outcome = miss_outcome(np.array([100.0, 50.0, 0.0]), 12.0)
    assert outcome.miss_m == 0.5
    assert outcome.timed_out is False
    assert outcome.time_s == 12.0

def test_timeout_has_no_miss():
    outcome = timeout_outcome(20.0)
    assert outcome.miss_m is None
    assert outcome.timed_out is True
    assert outcome.time_s == 20.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_outcome.py -v`

Expected: FAIL with import error for `rotorpy.guidance.outcome`.

- [ ] **Step 3: Implement `HomingOutcome`, `miss_outcome`, and `timeout_outcome` in `rotorpy/guidance/outcome.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_outcome.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 7: Camera geometry

**Files:**
- Create: `rotorpy/guidance/camera_geometry.py`
- Test: `tests/test_homing_camera_geometry.py`

**Interfaces:**
- Consumes: nothing. This module does not import the other guidance modules, so it stays in Wave 1.
- Produces:
  - `square_resolution() -> Tuple[int, int]` returning `(2236, 2236)`
  - `pinhole_intrinsics(width: int, height: int, fov_x_deg: float, fov_y_deg: float) -> dict` with keys `fx`, `fy`, `width`, `height`, `cx`, `cy`, `dist_coeffs`. `cx = width / 2`, `cy = height / 2`, `dist_coeffs` is five zeros. `fx = (width / 2) / tan(radians(fov_x_deg) / 2)`, and the same for `fy` with height and `fov_y_deg`.
  - `boresight_frd(pitch_deg: float) -> np.ndarray`, `(cos(pitch), 0, sin(pitch))` for a positive pitch looking down
  - `pitched_extrinsics(pitch_deg: float) -> dict` with `position` zeros and `orientation` the body-to-camera quaternion `[i, j, k, w]` in the simulator body frame
  - `pixel_to_camera_ray(u: float, v: float, intrinsics: dict) -> np.ndarray`, the unit camera-frame direction. The camera looks along +z, with +x right and +y down.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.guidance.camera_geometry import (
    boresight_frd, pinhole_intrinsics, pitched_extrinsics, pixel_to_camera_ray, square_resolution)
from rotorpy.sensors.camera import PinholeCamera
from rotorpy.world import World

def test_five_megapixel_square_and_focal_length():
    assert square_resolution() == (2236, 2236)
    intrinsics = pinhole_intrinsics(2236, 2236, 90.0, 90.0)
    assert np.isclose(intrinsics["fx"], 1118.0)
    assert np.isclose(intrinsics["fy"], 1118.0)
    assert np.allclose(intrinsics["dist_coeffs"], 0.0)

def test_boresight_is_ten_degrees_down_in_frd():
    pitch = np.deg2rad(10.0)
    assert np.allclose(boresight_frd(10.0), [np.cos(pitch), 0.0, np.sin(pitch)])

def test_principal_point_ray_is_camera_z():
    intrinsics = pinhole_intrinsics(32, 32, 90.0, 90.0)
    ray = pixel_to_camera_ray(intrinsics["cx"], intrinsics["cy"], intrinsics)
    assert np.allclose(ray, [0.0, 0.0, 1.0])

def test_level_body_projects_boresight_to_the_principal_point():
    intrinsics = pinhole_intrinsics(32, 32, 90.0, 90.0)
    extrinsics = pitched_extrinsics(10.0)
    camera = PinholeCamera(intrinsics=intrinsics, extrinsics=extrinsics, splat_radius=1, feature_output="rgb")
    pitch = np.deg2rad(10.0)
    point = 10.0 * np.array([np.cos(pitch), 0.0, -np.sin(pitch)])
    world = World({
        "bounds": {"extents": (-20, 20, -20, 20, -20, 20)},
        "blocks": [],
        "features": {"points": [point.tolist()], "colors": [[1.0, 0.0, 0.0]]},
    })
    state = {"x": np.zeros(3), "q": np.array([0.0, 0.0, 0.0, 1.0])}
    measurement = camera.measurement(state, world)
    assert measurement["keypoints"].shape[0] == 1
    assert np.allclose(measurement["keypoints"][0], [intrinsics["cx"], intrinsics["cy"]], atol=1.0)
    assert np.allclose(extrinsics["position"], 0.0)
    Rotation.from_quat(extrinsics["orientation"])  # quaternion parses
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_camera_geometry.py -v`

Expected: FAIL with import error for `rotorpy.guidance.camera_geometry`.

- [ ] **Step 3: Implement the functions in `rotorpy/guidance/camera_geometry.py`**

`pitched_extrinsics` builds the body-to-camera rotation so the simulator-body boresight `(cos(pitch), 0, -sin(pitch))` maps to camera +z. Camera +x is body right (simulator body −y) and camera +y is body down (simulator body −z), adjusted by the same pitch so the three axes stay orthonormal. Do not import `rotorpy.guidance.frames`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_camera_geometry.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 8: Homing command

**Files:**
- Create: `rotorpy/guidance/homing.py`
- Test: `tests/test_homing_command.py`

**Interfaces:**
- Consumes: `sim_to_ned`, `ned_to_sim`, `sim_body_to_frd`, `target_pixel_count`, `target_centroid`, `ground_relative_position`, `pn_acceleration`, `desired_velocity`, `pixel_to_camera_ray`, and the rotation in `pitched_extrinsics`.
- Produces:
  - `body_frd_velocity(velocity_ned: np.ndarray, attitude_sim_quat: np.ndarray) -> np.ndarray`. At the identity quaternion, NED `(vn, ve, vd)` becomes FRD `(vn, -ve, vd)`.
  - `HomingCommand` with `velocity_ned`, `acceleration_ned`, and `velocity_body_frd` as shape `(3,)`, plus `yaw_rad: float`, `locked: bool`, `tracking: bool`.
  - `HomingGuidance(nav_constant: float, lock_pixels: int, initial_velocity_ned: np.ndarray, intrinsics: dict, extrinsics: dict)`.
  - `HomingGuidance.update(t: float, height_m: float, attitude_sim_quat: np.ndarray, velocity_ned: np.ndarray, image: Optional[np.ndarray]) -> HomingCommand`.
  - `flat_output(position_sim: np.ndarray, command: HomingCommand) -> dict` with keys `x`, `x_dot`, `x_ddot`, `x_dddot`, `x_ddddot`, `yaw`, `yaw_dot`.

`image` is `None` when this sim step has no new frame. Pixel count >= `lock_pixels` latches `locked` for the rest of the flight. A latched flight whose new frame has no ground sightline, or whose pixel count drops below `lock_pixels`, sets `tracking` false and holds the last desired NED velocity, the last yaw, and zero acceleration. A later frame that sees the target does not resume tracking. Before any ground sightline exists, the held velocity is `initial_velocity_ned`.

A frame that produces a ground sightline sets `tracking` true. Desired NED velocity is `desired_velocity(relative_position)`. Acceleration is `pn_acceleration` with `dt` equal to the time since the previous image, and `previous_sightline` `None` on the first image, so the first tracking command has zero acceleration. Yaw is `atan2(sightline_east, sightline_north)`.

`velocity_body_frd` is `body_frd_velocity(velocity_ned, attitude_sim_quat)`.

`flat_output` sets `x` to `position_sim`, `x_dot` to `ned_to_sim(velocity_ned)`, `x_ddot` to `ned_to_sim(acceleration_ned)`, jerk and snap to zeros, `yaw` to `yaw_rad`, and `yaw_dot` to 0.

Sightline chain for a centroid: camera ray, then the inverse of the extrinsics rotation into the simulator body, then `sim_body_to_frd`, then the attitude rotation into the simulator world, then `sim_to_ned`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.camera_geometry import boresight_frd, pitched_extrinsics
from rotorpy.guidance.frames import ned_to_sim
from rotorpy.guidance.homing import HomingGuidance, body_frd_velocity, flat_output
from rotorpy.guidance.sightline import ground_relative_position
from rotorpy.guidance.speed import desired_velocity

IDENTITY = np.array([0.0, 0.0, 0.0, 1.0])
INTRINSICS = {"fx": 16.0, "fy": 16.0, "width": 32, "height": 32,
              "cx": 15.5, "cy": 15.5, "dist_coeffs": np.zeros(5)}

def _course():
    direction = np.array([100.0, 50.0, 0.0])
    return 10.0 * direction / np.linalg.norm(direction)

def _blob(row0, col0):
    image = np.zeros((32, 32, 3))
    image[row0:row0 + 6, col0:col0 + 6, 0] = 1.0
    return image

def _guidance():
    return HomingGuidance(3.0, 36, _course(), INTRINSICS, pitched_extrinsics(10.0))

def test_identity_attitude_frd_velocity():
    assert np.allclose(body_frd_velocity(np.array([4.0, 5.0, 6.0]), IDENTITY), [4.0, -5.0, 6.0])

def test_coast_below_lock_and_flat_output():
    guidance = _guidance()
    command = guidance.update(0.0, 30.0, IDENTITY, _course(), np.zeros((32, 32, 3)))
    assert command.locked is False and command.tracking is False
    assert np.allclose(command.velocity_ned, _course())
    assert np.allclose(command.acceleration_ned, 0.0)
    short = _blob(13, 13).copy()
    short[13, 13] = 0.0  # 35 red pixels
    command = guidance.update(0.05, 30.0, IDENTITY, _course(), short)
    assert command.locked is False
    flat = flat_output(np.array([1.0, 2.0, 3.0]), command)
    assert np.allclose(flat["x"], [1.0, 2.0, 3.0])
    assert np.allclose(flat["x_dot"], ned_to_sim(command.velocity_ned))
    assert flat["yaw_dot"] == 0.0
    assert np.allclose(flat["x_dddot"], 0.0) and np.allclose(flat["x_ddddot"], 0.0)

def test_lock_tracks_boresight_then_holds_through_a_later_sighting():
    guidance = _guidance()
    centered = _blob(13, 13)  # centroid (15.5, 15.5), the principal point
    first = guidance.update(0.0, 30.0, IDENTITY, _course(), centered)
    expected = desired_velocity(ground_relative_position(30.0, boresight_frd(10.0)))
    assert first.locked is True and first.tracking is True
    assert np.allclose(first.acceleration_ned, 0.0)
    assert np.allclose(first.velocity_ned, expected)
    assert np.allclose(first.velocity_body_frd, body_frd_velocity(expected, IDENTITY))
    held = guidance.update(0.01, 30.0, IDENTITY, _course(), None)
    assert np.allclose(held.velocity_ned, first.velocity_ned)
    second = guidance.update(0.05, 30.0, IDENTITY, _course(), _blob(17, 13))
    assert second.tracking is True
    assert not np.allclose(second.acceleration_ned, 0.0)
    assert abs(np.dot(second.acceleration_ned, second.velocity_ned)) < 1e-8
    lost = guidance.update(0.10, 30.0, IDENTITY, _course(), np.zeros((32, 32, 3)))
    assert lost.locked is True and lost.tracking is False
    assert np.allclose(lost.velocity_ned, second.velocity_ned)
    assert np.allclose(lost.acceleration_ned, 0.0)
    assert lost.yaw_rad == second.yaw_rad
    again = guidance.update(0.15, 30.0, IDENTITY, _course(), centered)
    assert np.allclose(again.velocity_ned, lost.velocity_ned)
    assert np.allclose(again.acceleration_ned, 0.0)
    assert again.yaw_rad == lost.yaw_rad

def test_pixels_above_the_horizon_lock_without_tracking():
    guidance = _guidance()
    command = guidance.update(0.0, 30.0, IDENTITY, _course(), _blob(0, 13))
    assert command.locked is True and command.tracking is False
    assert np.allclose(command.velocity_ned, _course())
    assert np.allclose(command.acceleration_ned, 0.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_command.py -v`

Expected: FAIL with import error for `rotorpy.guidance.homing`.

- [ ] **Step 3: Implement `body_frd_velocity`, `HomingCommand`, `HomingGuidance`, and `flat_output` in `rotorpy/guidance/homing.py`**

Store the last image time and the last unit sightline only when `image` is not `None`. A call with `image is None` returns the command from the stored sightline (or the coast command) and does not replace that sightline.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_command.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 9: Nominal scene

**Files:**
- Create: `rotorpy/guidance/scene.py`
- Test: `tests/test_homing_scene.py`

**Interfaces:**
- Consumes: `ned_to_sim`, `square_resolution`, `pinhole_intrinsics`, `pitched_extrinsics`, `target_pixel_count`.
- Produces:
  - `initial_velocity_ned(start_ned: np.ndarray, target_ground_ned: np.ndarray, speed: float) -> np.ndarray`. Horizontal, zero NED down, magnitude `speed`, pointed from start xy to target xy.
  - `hover_rotor_speed(quad_params: dict) -> float`. One rotor. `4 * k_eta * speed**2` equals `mass * 9.81`.
  - `nominal_initial_state() -> dict` with keys `x`, `v`, `q`, `w`, `wind`, `rotor_speeds`. Position is `ned_to_sim((0, 0, -30))`. Velocity is `ned_to_sim` of the 10 m/s initial course toward `(100, 50, 0)`. Quaternion is a level yaw `atan2(50, 100)` about simulator z, `[i, j, k, w]`. `w` and `wind` are zeros. `rotor_speeds` is four copies of `hover_rotor_speed` for the Hummingbird params.
  - `nominal_world() -> World`. One red block, simulator extents `(99.75, 100.25, 49.75, 50.25, 0.0, 1.0)`, bounds `(-50, 300, -50, 200, -20, 80)`. Regular face features, `descriptor_noise` 0, color `[1, 0, 0]`.
  - `nominal_camera() -> PinholeCamera` at 2236×2236, 90° by 90°, `pitched_extrinsics(10)`, `feature_output="rgb"`, `frame_rate` 20.

Spacing and splat radius are whatever values make the render test pass. Start from spacing 0.05 m and splat radius 1, and change only those two if the count is below 36.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.pixels import target_pixel_count
from rotorpy.guidance.scene import hover_rotor_speed, initial_velocity_ned, nominal_camera, nominal_initial_state, nominal_world
from rotorpy.vehicles.hummingbird_params import quad_params

def test_initial_course_and_hover_thrust():
    velocity = initial_velocity_ned(np.array([0.0, 0.0, -30.0]), np.array([100.0, 50.0, 0.0]), 10.0)
    assert velocity[2] == 0.0
    assert np.isclose(np.linalg.norm(velocity), 10.0)
    assert velocity[0] > 0.0 and velocity[1] > 0.0
    omega = hover_rotor_speed(quad_params)
    assert np.isclose(4.0 * quad_params["k_eta"] * omega**2, quad_params["mass"] * 9.81)

def test_nominal_start_sees_a_stable_target_of_at_least_36_pixels():
    world = nominal_world()
    camera = nominal_camera()
    state = nominal_initial_state()
    first = target_pixel_count(camera.measurement(state, world)["image"])
    second = target_pixel_count(camera.measurement(state, world)["image"])
    assert first == second
    assert first >= 36
    assert np.allclose(state["x"], [0.0, 0.0, 30.0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_scene.py::test_initial_course_and_hover_thrust -v`

Expected: FAIL with import error for `rotorpy.guidance.scene`.

- [ ] **Step 3: Implement the scene functions in `rotorpy/guidance/scene.py`**

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_scene.py -v`

Expected: PASS. The render test builds one 2236×2236 image twice.

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 10: Flight runner

**Files:**
- Create: `rotorpy/guidance/runner.py`
- Test: `tests/test_homing_runner.py`

**Interfaces:**
- Consumes: `HomingGuidance`, `flat_output`, `nominal_initial_state`, `nominal_world`, `nominal_camera`, `initial_velocity_ned`, `pinhole_intrinsics`, `pitched_extrinsics`, `square_resolution`, `miss_outcome`, `timeout_outcome`, `sim_to_ned`, `SE3Control`, `Multirotor`, Hummingbird `quad_params`.
- Produces:
  - `flight_status(t: float, locked: bool, sim_z: float, sim_velocity: np.ndarray) -> Optional[str]`. Returns `"ground"` when `locked` and `sim_z <= 0`. Else `"timeout"` when any `sim_velocity` component is `> 20`, or when `t >= 20` and not `locked`, or when `t >= 120` and `locked`. Else `None`. Ground wins over over-speed.
  - `run_homing(nav_constant: float = 3.0, lock_pixels: int = 36) -> HomingOutcome`.
  - `sweep(nav_constants=(2, 3, 4, 5)) -> list` of `(nav_constant, HomingOutcome)`.
  - `format_nominal(outcome: HomingOutcome) -> str`, `miss 0.500` with three decimal places, or `timeout`.
  - `format_sweep_line(nav_constant: float, outcome: HomingOutcome) -> str`, `2 miss 0.500` or `2 timeout`.

`run_homing` builds the nominal scene, a `HomingGuidance` with the nominal 10 m/s course and the nominal camera intrinsics and extrinsics, an `SE3Control`, and a `Multirotor(..., control_abstraction="cmd_motor_speeds", aero=True, enable_ground=True)`. Sim step is 0.01 s. Render and call `update` at `t = 0` and whenever `t` is within `1e-9` of a multiple of 0.05 s; on other steps call `update` with `image=None`. Latch `locked` once a command has `locked` true. After `vehicle.step`, advance `t`, then loop. `flight_status` is evaluated on the state about to be used, including the initial state, before rendering. `"ground"` returns `miss_outcome(sim_to_ned(position), t)`. `"timeout"` returns `timeout_outcome(t)`.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from rotorpy.guidance.outcome import HomingOutcome
from rotorpy.guidance.runner import flight_status, format_nominal, format_sweep_line

def test_stop_rules():
    assert flight_status(20.0, False, 30.0, np.zeros(3)) == "timeout"
    assert flight_status(19.99, False, 30.0, np.zeros(3)) is None
    assert flight_status(0.0, False, 0.0, np.zeros(3)) is None
    assert flight_status(5.0, True, 0.0, np.zeros(3)) == "ground"
    assert flight_status(5.0, True, 0.01, np.zeros(3)) is None
    assert flight_status(120.0, True, 10.0, np.zeros(3)) == "timeout"
    assert flight_status(119.0, True, 10.0, np.zeros(3)) is None
    assert flight_status(0.0, False, 30.0, np.array([20.0, 0.0, 0.0])) is None
    assert flight_status(0.0, False, 30.0, np.array([20.001, 0.0, 0.0])) == "timeout"
    assert flight_status(5.0, True, 0.0, np.array([25.0, 0.0, 0.0])) == "ground"

def test_print_lines():
    miss = HomingOutcome(miss_m=0.5, timed_out=False, time_s=11.0)
    timeout = HomingOutcome(miss_m=None, timed_out=True, time_s=20.0)
    assert format_nominal(miss) == "miss 0.500"
    assert format_nominal(timeout) == "timeout"
    assert format_sweep_line(2, miss) == "2 miss 0.500"
    assert format_sweep_line(4, timeout) == "4 timeout"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_homing_runner.py -v`

Expected: FAIL with import error for `rotorpy.guidance.runner`.

- [ ] **Step 3: Implement `flight_status`, `run_homing`, `sweep`, and the two formatters in `rotorpy/guidance/runner.py`**

Height passed to the guidance is the simulator z. Velocity passed to the guidance is `sim_to_ned(state["v"])`. The flat-output position is `state["x"]` before the step. Do not call `safety_exit`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_homing_runner.py -v`

Expected: PASS

- [ ] **Step 5: Leave the work uncommitted**

---

### Task 11: Example and end-to-end sweep

**Files:**
- Create: `examples/camera_homing.py`
- Create: `tests/test_homing_e2e.py`
- Modify: `pyproject.toml` pytest `addopts` and `markers`
- Modify: `UPDATES.md`
- Modify: `README.md` (the v3.0 camera paragraph that lists the camera examples)

**Interfaces:**
- Consumes: `run_homing`, `sweep`, `format_nominal`, `format_sweep_line`
- Produces: a script whose default run prints one nominal line and exits. `python examples/camera_homing.py --sweep` prints four lines, one per constant 2, 3, 4, 5. No plots and no animation on either path. The flight stays under `if __name__ == "__main__"`.

- [ ] **Step 1: Write the failing test**

```python
import pathlib
import pytest
from rotorpy.guidance.runner import format_sweep_line, sweep, run_homing

def test_default_suite_skips_homing_e2e():
    text = pathlib.Path("pyproject.toml").read_text()
    assert "not homing_e2e" in text
    assert "homing_e2e:" in text

@pytest.mark.homing_e2e
def test_never_lock_times_out_at_20_seconds():
    outcome = run_homing(lock_pixels=10**9)
    assert outcome.timed_out is True
    assert outcome.miss_m is None
    assert outcome.time_s == pytest.approx(20.0, abs=0.011)

@pytest.mark.homing_e2e
def test_sweep_prints_one_outcome_per_constant(capsys):
    results = sweep((2, 3, 4, 5))
    assert [n for n, _ in results] == [2, 3, 4, 5]
    for nav_constant, outcome in results:
        assert outcome.timed_out or outcome.miss_m >= 0.0
        print(format_sweep_line(nav_constant, outcome))
    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert lines == [format_sweep_line(n, outcome) for n, outcome in results]
```

- [ ] **Step 2: Run the fast test to verify it fails**

Run: `pytest tests/test_homing_e2e.py::test_default_suite_skips_homing_e2e -v`

Expected: FAIL because `pyproject.toml` does not yet mention `homing_e2e`.

- [ ] **Step 3: Implement `examples/camera_homing.py` and the docs**

The script parses `--sweep`. Default calls `run_homing()` and prints `format_nominal`. `--sweep` calls `sweep((2, 3, 4, 5))` and prints `format_sweep_line` for each. In `pyproject.toml`, add marker `homing_e2e: full Hummingbird camera-homing flights` and change addopts to `-m "not run_examples and not homing_e2e"`. Add `UPDATES.md` section `3.1.0 - Camera homing example` at the top, with one line for the example and one line for the navigation-constant sweep. In the README camera paragraph, name `examples/camera_homing.py` next to the other camera examples.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_homing_e2e.py -v -m homing_e2e`

Expected: PASS. The never-lock flight renders at 20 Hz for 20 s at 2236×2236. The sweep runs four more flights. This takes minutes.

Run: `pytest tests/test_homing_frames.py tests/test_homing_pixels.py tests/test_homing_sightline.py tests/test_homing_navigation.py tests/test_homing_speed.py tests/test_homing_outcome.py tests/test_homing_camera_geometry.py tests/test_homing_command.py tests/test_homing_scene.py tests/test_homing_runner.py -v`

Expected: PASS, and the default suite does not collect the end-to-end file’s tests.

Run: `pytest tests/test_examples.py::test_example_syntax -q`

Expected: PASS, including `camera_homing.py`.

- [ ] **Step 5: Leave the work uncommitted**
