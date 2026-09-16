from dataclasses import dataclass
import numpy as np

@dataclass
class TrueState:
    timestamp: float
    position: np.ndarray      # [x, y, z] in meters
    velocity: np.ndarray      # [vx, vy, vz] in m/s
    acceleration: np.ndarray  # [ax, ay, az] in m/s^2

@dataclass
class IMUReading:
    timestamp: float
    accel: np.ndarray         # Measured [ax, ay, az] (m/s^2)
    gyro: np.ndarray          # Measured [gx, gy, gz] (rad/s)

@dataclass
class GPSReading:
    timestamp: float
    position: np.ndarray      # Measured [x, y, z] (meters)
    velocity: np.ndarray      # Measured [vx, vy, vz] (m/s)
    is_spoofed: bool = False
    