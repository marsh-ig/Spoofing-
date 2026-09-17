"""
Adversarial GPS Spoofing Attack Suite:
1. Abrupt Step Spoofing (Baseline)
2. Slam-Frog Stealthy Quadratic Drift (Sub-noise floor)
"""
import numpy as np
from src.models.data_types import GPSReading

class GPSSpoofer:
    def __init__(self, attack_type: str = "slam_frog", start_time: float = 25.0):
        self.attack_type = attack_type
        self.start_time = start_time
        # Subtle false acceleration capped below IMU 3-sigma noise floor (0.08 m/s^2)
        self.drift_accel = np.array([0.08, 0.06, 0.0]) # Drift along North-East
        self.abrupt_offset = np.array([30.0, 25.0, 0.0]) # Instant 30m jump

    def inject_spoofing(self, authentic_gps: GPSReading) -> GPSReading:
        t = authentic_gps.timestamp
        
        # If attack hasn't started yet, output clean authentic GPS
        if t < self.start_time:
            return authentic_gps

        dt_attack = t - self.start_time
        
        if self.attack_type == "abrupt":
            # Instant position jump
            spoofed_pos = authentic_gps.position + self.abrupt_offset
            spoofed_vel = authentic_gps.velocity.copy()
            
        elif self.attack_type == "slam_frog":
            # Quadratic sub-threshold position drift: Delta_p = 0.5 * a_fake * t^2
            # Linear velocity drift: Delta_v = a_fake * t
            pos_offset = 0.5 * self.drift_accel * (dt_attack ** 2)
            vel_offset = self.drift_accel * dt_attack
            
            spoofed_pos = authentic_gps.position + pos_offset
            spoofed_vel = authentic_gps.velocity + vel_offset
        else:
            return authentic_gps
            
        return GPSReading(
            timestamp=t,
            position=spoofed_pos,
            velocity=spoofed_vel,
            is_spoofed=True
        )