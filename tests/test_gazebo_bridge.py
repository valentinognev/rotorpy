import numpy as np
from scipy.spatial.transform import Rotation

from rotorpy.guidance.gazebo_bridge import (
    attitude_gazebo_to_homing,
    gazebo_position_to_sim,
    velocity_gazebo_to_ned,
)


def test_nose_north_is_the_homing_identity_yaw():
    gazebo = Rotation.from_euler("z", np.pi / 2).as_quat()
    homing = attitude_gazebo_to_homing(gazebo)
    assert np.allclose(Rotation.from_quat(homing).as_euler("xyz"), 0.0, atol=1e-8)


def test_nose_east_is_a_quarter_turn_toward_east():
    gazebo = Rotation.from_euler("z", 0.0).as_quat()
    homing = Rotation.from_quat(attitude_gazebo_to_homing(gazebo))
    assert np.allclose(homing.as_euler("xyz", degrees=True), [0.0, 0.0, 90.0])


def test_nose_up_stays_nose_up_in_the_homing_frame():
    gazebo = (Rotation.from_euler("z", np.pi / 2) * Rotation.from_euler("y", np.deg2rad(-10.0))).as_quat()
    forward = Rotation.from_quat(attitude_gazebo_to_homing(gazebo)).apply([1.0, 0.0, 0.0])
    assert forward[2] > 0.0
    assert abs(forward[1]) < 1e-8


def test_gazebo_position_is_north_east_up():
    assert np.allclose(gazebo_position_to_sim([4.0, 8.0, 3.0]), [8.0, 4.0, 3.0])


def test_enu_velocity_becomes_ned():
    ned = velocity_gazebo_to_ned(np.array([4.0, 8.0, -1.0]))
    assert np.allclose(ned, [8.0, 4.0, 1.0])
