# Upward camera for the homing close

## Intent

The forward camera loses the target when the nose drops, and the homing law then gives up. A second camera, aimed almost straight up from the body, is there so that the same drop swings the target into view. Once that picture contains the target, the drone homes from it and can finish the intercept.

## Goals

- A second camera with the same lens as the forward camera, aimed 70° above the nose.
- Homing uses the center of a square drawn around the target in the camera that currently has it.
- The upward camera can take over after the forward camera has already lost the target.
- After the upward camera itself loses the target, the drone flies straight along its last commanded direction and does not lock again.
- The flight ends when the vehicle meets the ground, or at 20 seconds, and for no other reason.
- The homing page shows both pictures, the square on the picture that is flying the drone, both camera frustums, the body forward-right-down frame, and the drone sketch.

## Non-goals

- Changing proportional navigation, the speed law, the SE3 controller, or the vehicle.
- A camera that is not at the body origin.
- Re-acquiring the target after the upward camera has lost it.
- Photoreal images, or a different field of view, resolution, or splat radius.

## Constraints

- Simulator body is forward-left-up. In `pitched_extrinsics`, positive pitch looks down. The forward camera stays at +10°. The upward camera is the same mount at −70°, so its optical axis is 70° above the body forward axis, toward body up.
- Both cameras use `nominal_camera`'s intrinsics: 2236×2236, 90° by 90° field of view, 20 Hz, splat radius 1, RGB features, position `[0, 0, 0]`.
- Target pixels stay `red > 0.5` and red strictly greater than green and blue.
- Lock threshold stays 36 target pixels.
- Simulation stays 100 Hz. Both cameras render on the existing 20 Hz ticks, including the tick at t = 0.
- Sightlines stay in NED, built the way `_sightline_ned` builds them today. The pixel that used to be the centroid is now the square center.
- Ground is simulator `z <= 0`. A ground stop reports `miss_outcome`. A 20 second stop reports `timeout_outcome`.
- The page stays a self-contained Plotly HTML file at `Results/camera_homing.html`.

## How the two cameras fly the drone

On ticks with no new frame, the command is unchanged. A missing image on a camera tick counts as zero target pixels.

A camera **sees** the target when its target-pixel count is at least the lock threshold. A camera **steers** on a tick when it sees the target and `ground_relative_position` along the square-center sightline is not `None`. The level upward camera looks 70° up through a 90° field of view, so every pixel in that picture is at least 25° above the body horizon. Steering from it requires the nose to have dropped far enough that the square center points at the ground.

The upward camera **latches** on the first tick where it sees the target, even when that sightline misses the ground. After the latch the forward camera never steers again, including on ticks where the forward camera also sees the target.

Before the latch, the forward camera steers on ticks where it can steer. Any other tick keeps the last commanded NED velocity, sets acceleration to zero, and is not a permanent loss. A later forward tick can steer again. A later upward tick can still latch.

After the latch, a tick where the upward camera sees the target but cannot steer keeps the last velocity and zero acceleration, and is not a permanent loss. A later upward tick that can steer does steer. The first steering tick from the upward camera resets the proportional-navigation memory, so that tick does not treat the jump from the previous sightline as a sightline rate.

After the latch, the next camera tick where the upward camera does not see the target is a permanent loss. Velocity stays at the last command, acceleration is zero, and every later frame is ignored.

A steering tick sets the velocity command to `desired_velocity` of the ground-relative position along that sightline, yaw to `atan2` of the sightline's north and east components, and the acceleration command to the existing proportional-navigation update. `locked` stays true once either camera has seen the target. `tracking` is true only on a steering tick. `source`, `tracker_center`, and `tracker_side` are set only on a steering tick.

Before either camera has seen the target, behavior matches today: the initial cruise velocity, zero acceleration, not locked.

This ground-miss rule is a call made for the plan. A sightline that misses the ground does not count as losing the upward camera, and it does not by itself steer.

## Tracker square

The square is the smallest axis-aligned square that contains every target pixel and shares the center of those pixels' bounding box.

For target pixels with inclusive column range `[col_min, col_max]` and row range `[row_min, row_max]`:

- `center_u = (col_min + col_max) / 2`
- `center_v = (row_min + row_max) / 2`
- `side = max(col_max - col_min + 1, row_max - row_min + 1)`

`center_u` and `center_v` are the pixel passed to the sightline, in the same column-row index space the centroid uses today. No square exists when the target-pixel count is zero. The square used for homing is computed on the full-resolution image of the tracking camera, not on the preview.

The page draws that square only on the preview of the camera that is tracking on that frame. Map full-resolution coordinates into the preview by dividing by the same integer step `preview_image` used. The overlay is a yellow square with a blue cross at its center. A coasting frame draws no square.

## When the flight ends

`flight_status(t, sim_z)` returns `"ground"` when `sim_z <= 0`, otherwise `"timeout"` when `t >= 20`, otherwise nothing. Ground is checked first. There is no speed limit and no extra time for a lock. The stopping sample is still recorded, including both camera pictures.

## What the page shows

Beside the 3D scene, the forward picture is on the left and the upward picture is on the right. The slider still selects a capture time and shows both pictures from that time.

The 3D scene draws the path, the target box, the body forward-right-down axes, the drone sketch (green forward arms, red rear arms, prop circles on the upper side), and two frustums. The forward frustum uses the +10° mount. The upward frustum uses the −70° mount. There is no second set of red-green-blue axes on either camera.

## Builder calls

- `nominal_upward_camera()` sits next to `nominal_camera()` and differs only in `pitched_extrinsics(-70)`.
- `HomingGuidance` takes the shared intrinsics plus both extrinsic dicts. `update` takes the forward image and the upward image, each of which may be `None`.
- `HomingCommand` gains `source` (`"forward"`, `"upward"`, or `None`), `tracker_center` (shape `(2,)` or `None`), and `tracker_side` (float or `None`). `source` is set only on a tick that is tracking.
- `HomingRecord.image` becomes two arrays, `forward_image` and `upward_image`, plus per-frame `source`, `tracker_center`, and `tracker_side` aligned with `image_time`, and one `preview_step` integer. Centers and `tracker_side` are full-resolution pixels. A frame that is not steering stores `source is None` and NaN center and side. The page divides center and side by `preview_step`.
- The old centroid helper can remain. Homing no longer calls it.
