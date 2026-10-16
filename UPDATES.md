# Updates

## 3.1.0 - Camera homing example
- `examples/camera_homing.py` prints one nominal camera-homing miss or timeout.
- `--sweep` repeats the flight for navigation constants 2, 3, 4, and 5.

## 3.0.2 - Writable acceleration array in wind filter example
- `examples/wind_filter_example.py` copies the acceleration array before subtracting thrust. Pandas 3 `to_numpy()` is read-only.

## 3.0.1 - NumPy 2 waypoint stacking
- `examples/multimav_multiprocessing_demo.py` uses `np.vstack` for MinSnap waypoints. `np.row_stack` was removed in NumPy 2.
