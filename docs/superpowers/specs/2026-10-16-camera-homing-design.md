# Camera homing on an AscTec Hummingbird

## Intent

A runnable example of a Hummingbird that finds a red box with a fixed camera and homes on it until it meets the ground. The score of a flight is the miss distance at ground contact. Repeating the flight while changing one homing parameter shows how that parameter moves the miss, which is how the navigation constant is chosen and later improved.

## Goals

- One example flight with the nominal parameters below, printing either the miss distance or a timeout.
- A sweep that repeats that flight for several navigation constants and prints each miss or timeout.
- The guidance uses the camera image, estimated height, and estimated attitude. The true target coordinates are used to place the box, to set the initial velocity direction, and to score the miss. They are not an input to the homing law after the flight has started.
- Unit tests cover the geometry and the guidance decisions without a full flight. An end-to-end test runs the sweep.

## Non-goals

- Wind, sensor-noise filters, learning, and photorealistic rendering.
- Changing the simulator’s over-speed exit or the Hummingbird parameter file.
- A pass/fail miss distance. No number was given for “close enough.” The sweep is the measurement.

## Scenario parameters

These are the nominal values. Each is a field in `examples/camera_homing.jsonc`, which `examples/camera_homing.py` reads for the flight and the sweep. `controller` in that file is `pid` (`Px4Cascade`, the shipped choice) or `geometric` (`SE3Control`).

| Parameter | Value |
|---|---|
| Vehicle | AscTec Hummingbird (`rotorpy/vehicles/hummingbird_params.py`) |
| Target ground point, NED | (100, 50, 0) m |
| Drone start, NED | (0, 0, −30) m |
| Initial velocity | 10 m/s, horizontal, pointed from the start toward the target ground point |
| Box | red, 0.5 m by 0.5 m footprint, 1 m tall, standing on the ground point |
| Camera mount | fixed to the body, at the body origin |
| Camera pitch | 10° down from the body forward axis, in the body FRD zx plane |
| Field of view | 90° horizontal by 90° vertical |
| Resolution | 5 megapixels |
| Lock threshold | 36 pixels of the target in the image |
| No-lock time limit | 20 s |

NED is north, east, down. The box occupies NED x [99.75, 100.25], y [49.75, 50.25], z [−1, 0]. Its volume center, used only for the miss, is NED (100, 50, −0.5).

## Success and termination

- Lock is the first frame in which the target covers at least 36 pixels.
- Before lock, the drone holds the initial 10 m/s horizontal course and does not steer.
- If lock never happens, the run stops at 20 s. The drone is still in the air. That result has no miss distance and is reported as a timeout.
- After lock, the run continues until the first sample at which the drone’s NED z is greater than or equal to 0 (the ground plane). The miss is the straight-line distance from the drone position at that sample to the box volume center.
- If the tracker loses the box after lock, the drone holds the last desired NED velocity, with the navigation acceleration at zero, until the ground sample. The miss is still scored there.
- A locked flight that has not reached the ground after 120 s is a timeout with no miss. This bound exists so a diverging run cannot hang the sweep. The same timeout report is used if the simulator’s over-speed guard stops the vehicle.

The box is visible to the camera. It does not end the flight. The stock collision exit would stop the vehicle on the box before the center of mass reached the ground, which would make the miss undefined on a direct hit.

## Guidance

The camera is fixed, so the 10° pitch never moves relative to the body. While the box is tracked:

1. The center of the target pixels is the direction to the target in the camera frame.
2. That direction is rotated into the body FRD frame, then into NED with the attitude estimate.
3. The relative target position is the point where that world sightline meets the ground plane. Height supplies the scale: if the drone’s height above the ground is h and the sightline’s down-component is positive, the range along the sightline is h divided by that down-component. A sightline that does not meet the ground in front of the drone is treated as a lost track.
4. The desired NED velocity has magnitude equal to the maximum available speed and points along that relative position.
5. That velocity is expressed in body FRD, and the desired acceleration is proportional navigation on the estimated line of sight: navigation constant times closing speed times the line-of-sight rate, perpendicular to the sightline. Closing speed is the component of the drone’s velocity toward the estimated target. The line-of-sight rate comes from successive estimated sightlines. On the first locked frame the rate is zero and the acceleration command is zero.
6. The commanded yaw puts the sightline in the body zx plane, so the target sits on the vertical centerline of the image. Pitch centering is only as good as the thrust attitude allows, because the camera cannot move independently of the body. The phrase for this is “as much as possible.”

The inner loop is the `controller` field in `examples/camera_homing.jsonc`. `pid` runs `Px4Cascade` on the velocity tier: the homing NED velocity and acceleration, with yaw `pi/2 - yaw_rad`. `geometric` runs `SE3Control`. Its flat output uses the current position as the position setpoint and the measured velocity as the velocity setpoint, so those gains do not fight the homing. `track_acceleration` carries the homing request inside the upright specific-force cone. Yaw is the centering yaw. Yaw rate is zero.

Height, attitude, and velocity fed to this law are the vehicle state. RotorPy’s default estimator returns an empty estimate, so there is no filter in this example. The guidance functions take height, attitude, and velocity as arguments so a later estimator can replace the true state without changing the law.

The navigation constant shipped with the example starts at 3. The sweep is how that choice is checked. The sweep values are 2, 3, 4, and 5.

## Frames

RotorPy’s world frame has z up (weight is along −z) and its body frame has thrust along +z. The example’s parameters and guidance are written in NED and body FRD.

- NED (n, e, d) converts to the simulator world as (n, e, −d).
- Body FRD (forward, right, down) converts to the simulator body (forward, left, up) by flipping the y and z components.

The camera optical axis, in body FRD, is 10° from +x toward +z: (cos 10°, 0, sin 10°). The image plane uses the existing pinhole camera: optical axis +z, x right, y down. Distortion coefficients are zero. With a level simulator body and the target placed on that optical axis, the target’s pixel is the principal point.

A 90° by 90° field of view on a square sensor gives focal lengths `fx = fy = (width / 2) / tan(45°)`. The 5 megapixel image is 2236 by 2236, which is the nearest square resolution to 5,000,000 pixels (4,999,696).

## Camera and the box

RotorPy draws a box by splatting colored feature points, not by filling polygons. The box is a world block with color `[1, 0, 0]` and face features on a regular grid, with no color noise, spaced tightly enough that the splats cover the faces. A target pixel is one whose red channel is above 0.5 and strictly greater than its green and blue channels. The pixel count is the number of those pixels. The target center is their centroid. That count is what the 36-pixel lock compares against.

The homing loop renders a frame, reads the count and the centroid, and drops the image. Frames are not stored. The simulator runs at 100 Hz. The camera and the homing update run at 20 Hz. Between frames the last count, centroid, and sightline are held.

## Maximum available speed

The Hummingbird’s thrust can sustain well above the speed at which `simulate` aborts. `safety_exit` stops the run when any velocity component exceeds 20 m/s. The maximum available speed in a commanded direction is the largest speed along that direction whose components all have absolute value 19 m/s or less. A horizontal or vertical command is therefore 19 m/s. A command with equal north and down components has magnitude 19√2 m/s, so each of those components is 19 m/s. The limit is recomputed whenever the commanded direction changes.

## What a run prints

The example prints one line for the nominal flight: `miss <meters>` or `timeout`.

The sweep prints one line per navigation constant, in the same form, for 2, 3, 4, and 5. Comparison is among the constants that reached the ground. A constant that never locked, or that hit the 120 s bound or the over-speed guard, is a timeout and has no miss.

## Tests

Unit tests, all without a full flight:

- NED and body-FRD conversions, including the start point (0, 0, −30) mapping to simulator (0, 0, 30).
- A level vehicle with the 10° camera: a point on the boresight projects to the image center, and the unprojection of the image center is that boresight.
- Ground-plane range from height and a downward sightline, and a lost-track result when the sightline does not meet the ground.
- Proportional-navigation acceleration is perpendicular to the sightline, zero on the first sample, and scales with the navigation constant and the closing speed.
- Desired speed respects the 19 m/s component cap in a horizontal direction and in a dive.
- Lock stays false below 36 target pixels and becomes true at 36. Before lock the commanded velocity is the initial 10 m/s course and the acceleration is zero.
- After a loss, the commanded velocity is the last desired velocity and the acceleration is zero.
- Miss is the distance from a ground-contact position to NED (100, 50, −0.5). A timeout result has no miss.

End-to-end, marked slow so the default suite does not run it:

- The nominal scene with the pixel threshold set above any possible image (for example 10⁹) returns a timeout at 20 s and no miss.
- The sweep over navigation constants 2, 3, 4, and 5 returns one outcome each, either a miss at a ground contact or a timeout, and prints them.

The end-to-end test does not assert a maximum miss. The printed misses are the record used to change the navigation constant.

## Left to the builder

- File layout, following the existing example and controller style, with the guidance math importable without running a flight.
- Hover rotor speed used in the initial state, computed from the Hummingbird mass, gravity, and `k_eta`.
- Feature spacing and splat radius that make the red faces read as a solid target at the nominal start, with a test that the resulting pixel count at that pose is stable across two renders.
- The exact finite-difference step for the line-of-sight rate.
- World bounds large enough for a 20 s coast at 10 m/s past the target, and for the dive.
- Whether the example opens the existing animation. The sweep used by the test does not animate and does not plot.
