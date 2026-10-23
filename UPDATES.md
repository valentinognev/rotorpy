# Updates

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
