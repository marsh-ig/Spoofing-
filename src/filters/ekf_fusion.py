"""
Loosely-Coupled INS/GNSS Extended Kalman Filter (EKF)
Fuses 100 Hz IMU propagation with 10 Hz GPS measurement corrections.
"""
import numpy as np
from src.models.data_types import IMUReading, GPSReading

class LooseCoupledEKF:
    def __init__(self, dt: float = 0.01):
        self.dt = dt
        # State vector: [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
        self.x = np.zeros(6)
        # Covariance matrix P (state uncertainty)
        self.P = np.eye(6) * 1.0
        
        # State transition matrix F
        self.F = np.eye(6)
        self.F[0:3, 3:6] = np.eye(3) * dt
        
        # Process noise covariance Q (IMU uncertainty)
        self.Q = np.eye(6) * 0.01
        self.Q[3:6, 3:6] *= 0.1
        
        # Measurement matrix H (GPS measures all 6 states)
        self.H = np.eye(6)
        
        # Measurement noise covariance R (GPS uncertainty)
        self.R = np.eye(6) * (1.2 ** 2)
        self.R[3:6, 3:6] = np.eye(3) * (0.1 ** 2)

    def initialize_state(self, initial_pos: np.ndarray, initial_vel: np.ndarray):
        self.x[0:3] = initial_pos
        self.x[3:6] = initial_vel

    def predict(self, imu: IMUReading):
        """100 Hz IMU state prediction step"""
        # Compensate for gravity on body Z
        accel_ned = imu.accel + np.array([0.0, 0.0, 9.81])
        
        # Kinematic state update
        self.x[0:3] += self.x[3:6] * self.dt + 0.5 * accel_ned * (self.dt ** 2)
        self.x[3:6] += accel_ned * self.dt
        
        # Covariance propagation: P = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update_gps(self, gps: GPSReading):
        """10 Hz GPS measurement correction step"""
        z = np.concatenate([gps.position, gps.velocity])
        
        # Innovation residual: gamma = z - H*x
        gamma = z - (self.H @ self.x)
        
        # Innovation covariance: S = H*P*H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman Gain: K = P * H^T * inv(S)
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update: x = x + K * gamma
        self.x += K @ gamma
        
        # Covariance update: P = (I - K*H) * P
        I = np.eye(6)
        self.P = (I - K @ self.H) @ self.P
        
        # MUST return both gamma and S
        return gamma, S