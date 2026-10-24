# Controller Module

Each controller is its own package in this folder.

The geometric tracking controller is `SE3Control` in `quadrotor_control/`. Based on [this paper](https://mathweb.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf) it takes flat outputs (position and yaw) and outputs a dictionary containing different control abstractions (e.g., angle, rate, motor speeds).

`px4_cascade/` is `Px4Cascade`, the PX4 position, velocity, attitude, and rate cascade. It outputs Hummingbird motor speeds. `raptor/` is the RAPTOR foundation policy. `controller_template/` is the empty controller to copy.

Camera homing picks the inner loop in `examples/camera_homing.jsonc`. `"controller": "geometric"` is `SE3Control`. `"controller": "pid"` is `Px4Cascade`, and that is the shipped choice.

Currently, the `Multirotor` object accepts the following controller inputs (abstractions): 
- `cmd_motor_speeds`: the lowest control abstraction. The controller directly commands individual motor speeds.
- `cmd_motor_thrusts`: one step up, the controller commands individual thrusts for each motor.
- `cmd_ctbr`: the controller commands a collective thrust (ct) and body rates (br) on each axis. 
- `cmd_ctbm`: the controller commands a collective thrust (ct) and body moments (bm) on each axis.
- `cmd_ctatt`: the controller commands a collective thrust (ct) and attitude (as a quaternion). 
- `cmd_vel`: the controller commands a velocity vector in the world frame. Assumes `yaw` is 0.

For higher control abstractions, e.g. `cmd_vel` or `cmd_ctatt`, the lower level controllers are hidden in `Multirotor`. The gains for these controllers were hand-tuned for the Crazyflie parameters, so they may need tuning if a different vehicle is being used. 

Other controllers can be developed but must complement the vehicle and the trajectory they are trying to stabilize to. 

**NEW in `v2.0.0`:** RotorPy now has a batched simulation environment. Controllers can (and should) be parallelized over multiple UAVs using PyTorch operations which can utilize GPU resources. For an example of the expected output of this new batched controller, see `BatchedSE3Control` in `quadrotor_control/`. 