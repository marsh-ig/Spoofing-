"""
Synthetic GNSS/GPS Receiver: Generates noisy 10 Hz position and velocity fixes.
"""
import numpy as np
from src.models.data_types import TrueState, GPSReading

class SyntheticGPS:
    def __init__(self, pos_noise_std: float = 1.2, vel_noise_std: float = 0.1):
        self.pos_noise_std = pos_noise_std
        self.vel_noise_std = vel_noise_std

    def measure(self, state: TrueState) -> GPSReading:
        pos_measured = state.position + np.random.normal(0, self.pos_noise_std, 3)
        vel_measured = state.velocity + np.random.normal(0, self.vel_noise_std, 3)
        
        return GPSReading(
            timestamp=state.timestamp,
            position=pos_measured,
            velocity=vel_measured,
            is_spoofed=False
        )