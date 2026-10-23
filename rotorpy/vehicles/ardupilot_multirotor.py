"""This class inherits from Multirotor and is used to interface with Ardupilot SITL.
It sends the state to the SITL to verify that the conversion is correct.
"""
import copy
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
from rotorpy.frames import FrameConvention, M_FLU_TO_FRD
from rotorpy.vehicles.multirotor import Multirotor
from rotorpy.sensors.imu import Imu
from rotorpy.vehicles.hummingbird_params import quad_params as crazyflie_params
from scipy.spatial.transform import Rotation as R

try:
    from ArduPilotPlugin import ArduPilotPlugin
except:
    print("Missing dependency ArduPilotPlugin. Install it from source at https://github.com/TomerTip/PyArduPilotPlugin")

PWM_MIN = 1100
PWM_MAX = 1900
SEND_RATE_HZ = 1000

@dataclass
class ControlCommand:
    cmd_motor_speeds: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
        
    def to_dict(self) -> dict:
        return asdict(self)

x0 = {'x': np.array([0,0,0]),
      'v': np.zeros(3,),
      'q': np.array([0, 0, 0, 1]), # [i,j,k,w]
      'w': np.zeros(3,),
      'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
      'rotor_speeds': np.array([0.0, 0.0, 0.0, 0.0])}


@dataclass
class SensorData:
    sim_position : List[float]
    sim_attitude : List[float]
    sim_velocity_inertial : List[float]
    xgyro: float
    ygyro: float
    zgyro: float
    xacc: float
    yacc: float 
    zacc: float
    
    def __post_init__(self):
        assert len(self.sim_position) == 3
        assert len(self.sim_velocity_inertial) == 3
        assert len(self.sim_attitude) == 4


def _as_vec3(vector: np.ndarray) -> np.ndarray:
    return np.array(vector, dtype=float, copy=True).reshape(3)


def _inverse_body_to_world_scalar_first(q_xyzw: np.ndarray) -> List[float]:
    """Inverse of a scalar-last body-to-world quaternion, packed scalar-first."""
    q = np.array(q_xyzw, dtype=float, copy=True).reshape(4)
    return [float(q[3]), float(-q[0]), float(-q[1]), float(-q[2])]


def _body_frd(frame: FrameConvention, vector: np.ndarray) -> np.ndarray:
    """FLU-ENU applies ``M_FLU_TO_FRD``. FRD-NED keeps the measured axes."""
    body = _as_vec3(vector)
    if frame == FrameConvention.FRD_NED:
        return body
    if frame == FrameConvention.FLU_ENU:
        return M_FLU_TO_FRD @ body
    raise ValueError("frame must be FLU_ENU or FRD_NED")


def ardupilot_sensor_fields(
    frame: FrameConvention,
    state: Dict[str, np.ndarray],
    accel_body: np.ndarray,
    gyro_body: np.ndarray,
) -> dict:
    """Pack ArduPilot sensor fields. Does not construct ``Ardupilot``.

    Position and velocity are ``state['x']`` and ``state['v']`` in both frames.
    FLU-ENU applies ``M_FLU_TO_FRD`` to accel and gyro and uses
    ``Ardupilot._quaternion_rotorpy_to_aerospace``. FRD-NED packs accel and gyro
    as given. The FRD quaternion is the inverse, scalar-first, with no axis sandwich.
    """
    if frame == FrameConvention.FRD_NED:
        attitude = _inverse_body_to_world_scalar_first(state["q"])
    elif frame == FrameConvention.FLU_ENU:
        attitude = Ardupilot._quaternion_rotorpy_to_aerospace(
            np.asarray(state["q"], dtype=float)
        )
    else:
        raise ValueError("frame must be FLU_ENU or FRD_NED")
    return {
        "sim_position": _as_vec3(state["x"]).tolist(),
        "sim_velocity_inertial": _as_vec3(state["v"]).tolist(),
        "sim_attitude": attitude,
        "accel": _body_frd(frame, accel_body),
        "gyro": _body_frd(frame, gyro_body),
    }


class Ardupilot(Multirotor):
    M_glu2frd = R.from_euler('x', np.pi)
    M_enu2ned = R.from_matrix([[0, 1, 0], [1, 0, 0], [0, 0, -1]])

    def __init__(self, quad_params, initial_state = x0, control_abstraction='cmd_motor_speeds', aero=True, ardupilot_control=False, enable_imu_noise = False, enable_ground = False):
        super().__init__(quad_params, initial_state, control_abstraction, aero, enable_ground)
        self._enable_ground = enable_ground
        self._ardupilot_control = ardupilot_control
        statedot = {'vdot' : np.zeros(3,), 'wdot' : np.zeros(3,)}
        self.t = 0.0
        self.imu = Imu()
        self._frame = FrameConvention.FLU_ENU
        self.__enable_imu_noise = enable_imu_noise

        self.sensor_data = self._create_sensor_data(
            self.initial_state, statedot, self.imu, self.__enable_imu_noise, self._frame
        )
        self.ap_link = ArduPilotPlugin()
        self.ap_link.drain_unread_packets()

        self._control_cmd = ControlCommand()   


    def step(self, state, control, t_step):
        received, pwm = self.ap_link.pre_update(self.t)
        if self._ardupilot_control:
            control = {'cmd_motor_speeds': self._motor_cmd_to_omega(self._control_cmd.cmd_motor_speeds)}


        statedot = self.statedot(state, control, t_step)
        state =  super().step(state, control, t_step)
        self.t += t_step

        self.sensor_data = self._create_sensor_data(
            state, statedot, self.imu, self.__enable_imu_noise, self._frame
        )
        
        self.ap_link.post_update(self.sensor_data, self.t)
        if received: # TODO: is this being handled correctly?
            self._control_cmd.cmd_motor_speeds = list(pwm[0:4]) # type: ignore
        
        return state


    @staticmethod
    def _motor_cmd_to_omega(pwm_commands : List[int]) -> List[float]:
        """Convert the pwm commands received from the SITL into angular velocity targets for the motors.

        Args:
            pwm_commands (List[int]): pwm commands [1000-2000] output by Ardupilot SITL

        Returns:
            List[float]: angular velocities targets for the motors.
        """
        rotor_0, rotor_1, rotor_2, rotor_3 = pwm_commands
        reordered_pwm_commands = [rotor_2, rotor_0, rotor_3, rotor_1]
        normalized_commands = [(c-PWM_MIN)/(PWM_MAX-PWM_MIN) for c in reordered_pwm_commands]
        angular_velocities = [838.0*c for c in normalized_commands] # TODO: remove magic constant
        return angular_velocities

    def set_frame(self, frame):
        """Remember the world frame and apply it to the simulated IMU."""
        super(Ardupilot, self).set_frame(frame)
        self._frame = FrameConvention(frame)
        self.imu.set_frame(self._frame)

    @staticmethod
    def _quaternion_rotorpy_to_aerospace(quaternion_glu2enu: np.ndarray) -> List[float]:
        """
        Convert quaternion from rotorpy convention to aerospace (ArduPilot) convention.
        
        Args:
            quaternion_glu2enu (np.ndarray): Quaternion [x, y, z, w] (scalar-last) representing 
                                           rotation from body frame (GLU) to world frame (ENU)
        
        Returns:
            List[float]: Quaternion [w, x, y, z] (scalar-first) representing rotation from 
                        world frame (NED) to body frame (FRD) - aerospace convention
        
        Notes:
            - Input: rotorpy uses scalar-last [x, y, z, w] for GLU→ENU rotation
            - Output: ArduPilot uses scalar-first [w, x, y, z] for NED→FRD rotation (inverse)
            - Involves coordinate frame transformations: GLU↔FRD and ENU↔NED
        """
        # Convert rotorpy quaternion (GLU→ENU) to rotation object
        R_glu2enu = R.from_quat(quaternion_glu2enu, scalar_first=False)
        
        # Transform to aerospace convention (NED→FRD)
        # R_frd2ned represents the attitude of the FRD frame in the NED frame
        R_frd2ned = Ardupilot.M_enu2ned * R_glu2enu * Ardupilot.M_glu2frd
        
        # Return as scalar-first quaternion [w, x, y, z]
        return R_frd2ned.as_quat(scalar_first=True).tolist()      

    def _create_sensor_data(
        self,
        state: Dict[str, np.ndarray],
        statedot: Dict[str, np.ndarray],
        imu: Imu,
        enable_imu_noise: bool = False,
        frame: Optional[FrameConvention] = None,
    ) -> SensorData:
        """
        Converts the state and state derivative of a `rotorpy.vehicles.multirotor` object to the Ardupilot convention
        for position, velocity, acceleration, quaternions, and angular velocities.
        Args:
            state (Dict[str, np.ndarray]): The current state of the multirotor, including position, velocity, and attitude quaternion.
            statedot (np.ndarray): The state derivative, representing the acceleration.
            imu (Imu): The IMU object used to obtain measurements.
            enable_imu_noise (bool, optional): Flag to enable or disable IMU noise in the measurements. Defaults to False.
            frame (FrameConvention, optional): Active world/body pair. Defaults to the frame stored by ``set_frame``
                (FLU-ENU until then). FLU-ENU keeps the historical axis rewrite. FRD-NED packs the measured accel and
                gyro and the inverse attitude.
        Notes:
            - The attitude quaternion in rotorpy is in scalar-last representation (x, y, z, w) and represents the rotation
              from the body frame to the world frame.
            - The attitude quaternion in Ardupilot is in scalar-first representation (w, x, y, z) and represents the rotation
              from the world frame (NED) to the body frame (FRD).

        Returns:
            SensorData: The sensor data in the Ardupilot convention, including position, velocity, acceleration,
                        attitude quaternion, and angular velocities.
        """

        if frame is None:
            frame = self._frame
        acceleration = copy.deepcopy(statedot)
        meas_dict = imu.measurement(state, acceleration, with_noise=enable_imu_noise)
        fields = ardupilot_sensor_fields(frame, state, meas_dict["accel"], meas_dict["gyro"])
        accel = fields["accel"]
        gyro = fields["gyro"]
        return SensorData(
            fields["sim_position"],
            fields["sim_attitude"],
            fields["sim_velocity_inertial"],
            xgyro=float(gyro[0]),
            ygyro=float(gyro[1]),
            zgyro=float(gyro[2]),
            xacc=float(accel[0]),
            yacc=float(accel[1]),
            zacc=float(accel[2]),
        )


if __name__ == '__main__':
    import time
    r = R.from_euler('y', 0, degrees=True)
    initial_state = {'x': np.array([0,0,0]),
                                            'v': np.zeros(3,),
                                            'q': r.as_quat(scalar_first=False), # [i,j,k,w]
                                            'w': np.zeros(3,),
                                            'wind': np.array([0,0,0]),  # Since wind is handled elsewhere, this value is overwritten
                                            'rotor_speeds': np.array([0, 0, 0, 0])}
    vehicle = Ardupilot(initial_state=initial_state, quad_params = crazyflie_params, ardupilot_control=True, enable_imu_noise=False, enable_ground=True)

    state = initial_state

    dt = 0.01
    while True:
        state = vehicle.step(state, {'cmd_motor_speeds': [0,]*4}, dt)
        time.sleep(dt)
