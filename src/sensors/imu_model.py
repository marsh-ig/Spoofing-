"""
Synthetic MEMS IMU: Generates 100 Hz specific forces and angular rates with noise and bias drift.
"""
import numpy as np
from src.models.data_types import TrueState, IMUReading

class SyntheticIMU:
    def __init__(self, accel_noise_std: float = 0.05, gyro_noise_std: float = 0.005):
        self.accel_noise_std = accel_noise_std
        self.gyro_noise_std = gyro_noise_std
        # Realistic constant sensor bias
        self.accel_bias = np.array([0.01, -0.01, 0.02])

    def measure(self, state: TrueState) -> IMUReading:
        # Gravity in NED frame (downward is positive Z)
        gravity = np.array([0.0, 0.0, -9.81])
        
        # Specific force = true kinematic acceleration - gravity + bias + Gaussian noise
        accel_body = state.acceleration - gravity + self.accel_bias
        accel_measured = accel_body + np.random.normal(0, self.accel_noise_std, 3)
        gyro_measured = np.random.normal(0, self.gyro_noise_std, 3)
        
        return IMUReading(
            timestamp=state.timestamp,
            accel=accel_measured,
            gyro=gyro_measured
        )