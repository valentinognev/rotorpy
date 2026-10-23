import numpy as np
from scipy.spatial.transform import Rotation
from rotorpy.frames import (
    M_ENU_TO_NED, M_FLU_TO_FRD, FrameConvention, aabb_enu_to_ned,
    apply_frame, attitude_enu_flu_to_ned_frd, attitude_ned_frd_to_enu_flu,
    body_to_sensor_flu_to_frd, enu_to_ned, flat_enu_to_ned, flu_to_frd,
    frd_to_flu, ned_to_enu, yaw_enu_to_ned, yaw_ned_to_enu, yaw_rate_enu_to_ned,
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


def test_vector_maps_copy_the_input():
    original = np.array([1.0, 2.0, 3.0])
    converted = enu_to_ned(original)
    converted[0] = 99.0
    assert np.allclose(original, [1.0, 2.0, 3.0])
    assert np.allclose(flu_to_frd(original), [1.0, -2.0, -3.0])


def test_yaw_ned_to_enu_undoes_the_enu_yaw():
    assert yaw_ned_to_enu(np.pi / 2) == 0.0
    assert yaw_ned_to_enu(0.0) == np.pi / 2


def test_flat_converts_higher_derivatives_and_keeps_other_keys():
    flat = {
        "x_dot": np.array([1.0, 2.0, 3.0]),
        "x_ddot": np.array([1.0, 0.0, 0.0]),
        "x_dddot": np.array([0.0, 1.0, 0.0]),
        "x_ddddot": np.array([0.0, 0.0, 1.0]),
        "yaw_ddot": 0.4,
    }
    converted = flat_enu_to_ned(flat)
    assert np.allclose(converted["x_dot"], [2.0, 1.0, -3.0])
    assert np.allclose(converted["x_ddot"], [0.0, 1.0, 0.0])
    assert np.allclose(converted["x_dddot"], [1.0, 0.0, 0.0])
    assert np.allclose(converted["x_ddddot"], [0.0, 0.0, -1.0])
    assert converted["yaw_ddot"] == 0.4
    assert flat["yaw_ddot"] == 0.4


def test_body_to_sensor_right_multiplies_by_flu_to_frd():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    rotation = Rotation.from_quat(body_to_sensor_flu_to_frd(q)).as_matrix()
    assert np.allclose(rotation, M_FLU_TO_FRD)


def test_apply_frame_calls_set_frame_only_on_objects_that_have_it():
    class _Framed(object):
        def __init__(self):
            self.seen = None

        def set_frame(self, frame):
            self.seen = frame

    class _Bare(object):
        pass

    framed = _Framed()
    bare = _Bare()
    assert apply_frame(FrameConvention.FRD_NED, framed, None, bare) is None
    assert framed.seen is FrameConvention.FRD_NED


def test_frame_convention_uses_the_pair_names():
    assert FrameConvention.FLU_ENU == "flu_enu"
    assert FrameConvention.FRD_NED == "frd_ned"
