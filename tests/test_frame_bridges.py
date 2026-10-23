import numpy as np
import pytest

from rotorpy.frames import FrameConvention
from rotorpy.sensors.imu import Imu
from rotorpy.vehicles.ardupilot_multirotor import Ardupilot
from rotorpy.vehicles.hummingbird_params import quad_params
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.vehicles.px4_multirotor import PX4Multirotor
import rotorpy.vehicles.ardupilot_multirotor as ardupilot_module
import rotorpy.vehicles.px4_multirotor as px4_module


def _identity_state(position, velocity):
    return {
        "x": np.array(position, dtype=float),
        "v": np.array(velocity, dtype=float),
        "q": np.array([0.0, 0.0, 0.0, 1.0]),
        "w": np.array([0.1, 0.2, 0.3]),
    }


def test_px4_frd_keeps_millig():
    """FRD packing scales milli-g without a second y/z negation.

    A FLU-style negation of accel [0, 0, -9.81] flips zacc_mg positive.
    Assigning gyro from state['w'] after a y/z flip fails the body-rate check.
    Scalar-first reordering without the attitude inverse fails the pitched quaternion.
    """
    gyro = np.array([0.4, 0.5, 0.6])
    fields = px4_module.px4_hil_fields(
        FrameConvention.FRD_NED,
        _identity_state([20.0, 10.0, -5.0], [2.0, 1.0, -3.0]),
        np.array([0.0, 0.0, -9.81]),
        gyro,
    )
    assert (fields["vx_cms"], fields["vy_cms"], fields["vz_cms"]) == (200, 100, -300)
    assert fields["zacc_mg"] == int(np.round(-9.81 / 9.80665 * 1000))
    assert (fields["rollspeed"], fields["pitchspeed"], fields["yawspeed"]) == (0.1, 0.2, 0.3)
    assert fields["quaternion"] == [1, 0, 0, 0]
    assert np.allclose(fields["gyro"], [0.4, 0.5, 0.6])

    lat_deg, lon_deg, alt_m = PX4Multirotor.enu_to_geodetic(10.0, 20.0, 5.0)
    assert fields["lat_deg"] == lat_deg
    assert fields["lon_deg"] == lon_deg
    assert fields["alt_m"] == alt_m
    assert fields["alt_m"] == 5.0

    y_fields = px4_module.px4_hil_fields(
        FrameConvention.FRD_NED,
        _identity_state([20.0, 10.0, -5.0], [2.0, 1.0, -3.0]),
        np.array([0.0, 1.0, -9.81]),
        gyro,
    )
    assert y_fields["yacc_mg"] == int(np.round(1.0 / 9.80665 * 1000))
    assert y_fields["zacc_mg"] == int(np.round(-9.81 / 9.80665 * 1000))

    pitched = _identity_state([20.0, 10.0, -5.0], [2.0, 1.0, -3.0])
    pitched["q"] = np.array([0.3, -0.1, 0.2, np.sqrt(1.0 - 0.3 ** 2 - 0.1 ** 2 - 0.2 ** 2)])
    pitched_fields = px4_module.px4_hil_fields(
        FrameConvention.FRD_NED,
        pitched,
        np.array([0.0, 0.0, -9.81]),
        gyro,
    )
    q = pitched["q"]
    assert pitched_fields["quaternion"] == [q[3], -q[0], -q[1], -q[2]]


def test_px4_flu_matches_current_rewrite():
    """FLU packing keeps today's ENU velocity, y/z negation, and aerospace quaternion.

    Skipping the y/z negation makes zacc_mg positive for accel [0, 0, 9.81].
    Leaving velocity in ENU order yields vx_cms 100 instead of 200.
    """
    state = _identity_state([10.0, 20.0, 5.0], [1.0, 2.0, 3.0])
    gyro = np.array([0.4, 0.5, 0.6])
    fields = px4_module.px4_hil_fields(
        FrameConvention.FLU_ENU,
        state,
        np.array([0.0, 0.0, 9.81]),
        gyro,
    )
    assert (fields["vx_cms"], fields["vy_cms"], fields["vz_cms"]) == (200, 100, -300)
    assert fields["zacc_mg"] == int(np.round(-9.81 / 9.80665 * 1000))
    assert fields["xacc_mg"] == 0
    assert fields["yacc_mg"] == 0
    assert (fields["rollspeed"], fields["pitchspeed"], fields["yawspeed"]) == (0.1, 0.2, 0.3)
    assert np.allclose(fields["gyro"], [0.4, -0.5, -0.6])
    assert fields["quaternion"] == Ardupilot._quaternion_rotorpy_to_aerospace(state["q"])

    lat_deg, lon_deg, alt_m = PX4Multirotor.enu_to_geodetic(10.0, 20.0, 5.0)
    assert fields["lat_deg"] == lat_deg
    assert fields["lon_deg"] == lon_deg
    assert fields["alt_m"] == alt_m

    y_fields = px4_module.px4_hil_fields(
        FrameConvention.FLU_ENU,
        state,
        np.array([0.0, 1.0, 9.81]),
        gyro,
    )
    assert y_fields["yacc_mg"] == int(np.round(-1.0 / 9.80665 * 1000))
    assert y_fields["xacc_mg"] == 0


def test_ardupilot_frd_skips_axis_rewrite():
    """FRD packing keeps measured accel and gyro and the inverse attitude.

    Applying M_FLU_TO_FRD to accel [1, 2, 3] yields [1, -2, -3].
    The aerospace axis sandwich on an identity quaternion is not [1, 0, 0, 0].
    Scalar-first reordering without the inverse fails the pitched quaternion.
    """
    state = _identity_state([4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
    accel = np.array([1.0, 2.0, 3.0])
    gyro = np.array([0.1, 0.2, 0.3])
    fields = ardupilot_module.ardupilot_sensor_fields(
        FrameConvention.FRD_NED,
        state,
        accel,
        gyro,
    )
    assert fields["sim_position"] == [4.0, 5.0, 6.0]
    assert fields["sim_velocity_inertial"] == [7.0, 8.0, 9.0]
    assert np.allclose(fields["accel"], [1.0, 2.0, 3.0])
    assert np.allclose(fields["gyro"], [0.1, 0.2, 0.3])
    assert tuple(np.shape(fields["accel"])) == (3,)
    assert tuple(np.shape(fields["gyro"])) == (3,)
    assert fields["sim_attitude"] == [1, 0, 0, 0]

    pitched = _identity_state([4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
    pitched["q"] = np.array([0.3, -0.1, 0.2, np.sqrt(1.0 - 0.3 ** 2 - 0.1 ** 2 - 0.2 ** 2)])
    pitched_fields = ardupilot_module.ardupilot_sensor_fields(
        FrameConvention.FRD_NED,
        pitched,
        accel,
        gyro,
    )
    q = pitched["q"]
    assert pitched_fields["sim_attitude"] == [q[3], -q[0], -q[1], -q[2]]

    flu = ardupilot_module.ardupilot_sensor_fields(
        FrameConvention.FLU_ENU,
        state,
        accel,
        gyro,
    )
    assert flu["sim_position"] == [4.0, 5.0, 6.0]
    assert flu["sim_velocity_inertial"] == [7.0, 8.0, 9.0]
    assert np.allclose(flu["accel"], [1.0, -2.0, -3.0])
    assert np.allclose(flu["gyro"], [0.1, -0.2, -0.3])
    assert tuple(np.shape(flu["accel"])) == (3,)
    assert tuple(np.shape(flu["gyro"])) == (3,)
    assert flu["sim_attitude"] == Ardupilot._quaternion_rotorpy_to_aerospace(state["q"])


def test_ardupilot_set_frame_packet_uses_stored_frame():
    """After set_frame(FRD_NED), the packet uses that frame and the switched IMU.

    Omitting the frame leaves the FLU rewrite: gyro y flips sign, and the
    aerospace sandwich of identity is not [1, 0, 0, 0]. Leaving the IMU in
    FLU makes zacc +9.81 for world acceleration [1, 2, 0]. A second
    set_frame must not flip the IMU gravity back.
    """
    vehicle = Ardupilot.__new__(Ardupilot)
    Multirotor.__init__(vehicle, quad_params)
    vehicle.imu = Imu()
    vehicle.set_frame(FrameConvention.FRD_NED)
    vehicle.set_frame(FrameConvention.FRD_NED)
    assert getattr(vehicle, "_frame", None) is FrameConvention.FRD_NED
    assert vehicle.imu.gravity_vector[2] == pytest.approx(9.81)

    state = _identity_state([4.0, 5.0, 6.0], [7.0, 8.0, 9.0])
    state["w"] = np.array([0.1, 0.2, 0.3])
    statedot = {"vdot": np.array([1.0, 2.0, 0.0]), "wdot": np.zeros(3)}
    packet = vehicle._create_sensor_data(state, statedot, vehicle.imu, False)

    assert packet.sim_position == [4.0, 5.0, 6.0]
    assert packet.sim_velocity_inertial == [7.0, 8.0, 9.0]
    assert packet.sim_attitude == [1.0, 0.0, 0.0, 0.0]
    assert packet.xacc == pytest.approx(1.0)
    assert packet.yacc == pytest.approx(2.0)
    assert packet.zacc == pytest.approx(-9.81)
    assert packet.xgyro == pytest.approx(0.1)
    assert packet.ygyro == pytest.approx(0.2)
    assert packet.zgyro == pytest.approx(0.3)
