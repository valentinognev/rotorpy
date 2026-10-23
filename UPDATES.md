# Updates

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
