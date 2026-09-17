"""
Advanced Spoofer Suite:
1. Non-Reactive Spoofer (replays pre-calculated smooth path)
2. Smart Reactive Spoofer (tracks UAV with RF transport delay tau)
"""
import numpy as np
from src.models.data_types import GPSReading

class ReactiveSpoofer:
    def __init__(self, mode="reactive", transport_lag_sec=0.20, drift_accel=0.08):
        self.mode = mode
        self.transport_lag = transport_lag_sec # 200 ms RF synthesis delay
        self.drift_accel = np.array([drift_accel, drift_accel*0.75, 0.0])
        self.history = []

    def inject_spoofing(self, authentic_gps: GPSReading) -> GPSReading:
        t = authentic_gps.timestamp
        self.history.append((t, authentic_gps.position.copy(), authentic_gps.velocity.copy()))
        
        # Attack starts after 10s
        if t < 10.0:
            return authentic_gps
            
        dt_attack = t - 10.0
        drift_offset = 0.5 * self.drift_accel * (dt_attack ** 2)
        vel_drift = self.drift_accel * dt_attack
        
        if self.mode == "non_reactive":
            # Smooth pre-calculated trajectory (completely misses micro-twitches)
            smooth_pos = authentic_gps.position[0] + np.array([20.0*t, 0.0, 0.0]) + drift_offset
            return GPSReading(t, smooth_pos, authentic_gps.velocity + vel_drift, is_spoofed=True)
            
        elif self.mode == "reactive":
            # Smart attacker tracks UAV, but delayed by RF transport lag (tau = 200ms)
            target_t = max(0.0, t - self.transport_lag)
            
            # Find the delayed historical state
            delayed_pos = authentic_gps.position
            delayed_vel = authentic_gps.velocity
            for h_t, h_pos, h_vel in reversed(self.history):
                if h_t <= target_t:
                    delayed_pos = h_pos
                    delayed_vel = h_vel
                    break
                    
            spoofed_pos = delayed_pos + drift_offset
            spoofed_vel = delayed_vel + vel_drift
            return GPSReading(t, spoofed_pos, spoofed_vel, is_spoofed=True)
            
        return authentic_gps