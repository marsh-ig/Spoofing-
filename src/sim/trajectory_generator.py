import numpy as np
import pandas as pd

def generate_flight(duration_sec: float = 60.0, dt: float = 0.01):
    time_steps = int(duration_sec / dt)
    records = []
    
    pos = np.array([0.0, 0.0, -100.0]) # 100m altitude
    vel = np.array([20.0, 0.0, 0.0])   # 20 m/s cruise
    heading = 0.0
    
    for step in range(time_steps):
        t = round(step * dt, 4)
        
        # Turn maneuver between 20s and 35s
        if 20.0 <= t <= 35.0:
            heading += np.radians(6.0) * dt
            vel = np.array([20.0 * np.cos(heading), 20.0 * np.sin(heading), 0.0])
        
        pos += vel * dt
        
        # 100 Hz IMU + Noise
        imu_accel = np.array([0.0, 0.0, -9.81]) + np.random.normal(0, 0.05, 3)
        
        # 10 Hz GPS (every 10 steps)
        is_gps = (step % 10 == 0)
        gps_pos = pos + np.random.normal(0, 1.5, 3) if is_gps else None
        
        records.append({
            'timestamp': t,
            'pos_x': pos[0], 'pos_y': pos[1], 'pos_z': -pos[2],
            'vel_x': vel[0], 'vel_y': vel[1], 'vel_z': vel[2],
            'imu_ax': imu_accel[0], 'imu_ay': imu_accel[1], 'imu_az': imu_accel[2],
            'gps_x': gps_pos[0] if is_gps else np.nan,
            'gps_y': gps_pos[1] if is_gps else np.nan,
            'gps_z': -gps_pos[2] if is_gps else np.nan,
        })
        
    return pd.DataFrame(records)
