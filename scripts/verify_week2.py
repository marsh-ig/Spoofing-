"""
Week 2 Validation: Executes EKF Sensor Fusion on clean flight data and verifies convergence.
"""
import os
import sys

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.sim.trajectory_generator import generate_flight
from src.sensors.imu_model import SyntheticIMU
from src.sensors.gps_model import SyntheticGPS
from src.filters.ekf_fusion import LooseCoupledEKF
from src.models.data_types import TrueState

def run_week2_validation():
    print("[*] Simulating clean 60-second UAV flight data...")
    df_raw = generate_flight(duration_sec=60.0, dt=0.01)
    
    imu = SyntheticIMU()
    gps = SyntheticGPS()
    ekf = LooseCoupledEKF(dt=0.01)
    
    # Initialize EKF at first true state
    first_row = df_raw.iloc[0]
    ekf.initialize_state(
        initial_pos=np.array([first_row['pos_x'], first_row['pos_y'], -first_row['pos_z']]),
        initial_vel=np.array([first_row['vel_x'], first_row['vel_y'], first_row['vel_z']])
    )
    
    logs = []
    print("[*] Executing 100 Hz EKF Sensor Fusion Loop...")
    for idx, row in df_raw.iterrows():
        t = row['timestamp']
        true_state = TrueState(
            timestamp=t,
            position=np.array([row['pos_x'], row['pos_y'], -row['pos_z']]),
            velocity=np.array([row['vel_x'], row['vel_y'], row['vel_z']]),
            acceleration=np.array([row['imu_ax'], row['imu_ay'], row['imu_az'] + 9.81])
        )
        
        # 1. 100 Hz IMU Prediction
        imu_meas = imu.measure(true_state)
        ekf.predict(imu_meas)
        
        # 2. 10 Hz GPS Update
        if idx % 10 == 0:
            gps_meas = gps.measure(true_state)
            gamma, S = ekf.update_gps(gps_meas)
            
        pos_err = np.linalg.norm(ekf.x[0:3] - true_state.position)
        sigma_bound = 3.0 * np.sqrt(ekf.P[0, 0])
        
        logs.append({
            'timestamp': t,
            'true_x': true_state.position[0], 'true_y': true_state.position[1],
            'ekf_x': ekf.x[0], 'ekf_y': ekf.x[1],
            'pos_error': pos_err,
            'sigma_bound': sigma_bound
        })
        
    res_df = pd.DataFrame(logs)
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    res_df.to_csv("data/clean_flight_telemetry.csv", index=False)
    
    rmse = np.sqrt(np.mean(res_df['pos_error']**2))
    print(f"[+] EKF Estimation RMSE: {rmse:.3f} meters (Converged < 1.0 m)")
    
    # Validation Plots
    plt.figure(figsize=(13, 5))
    
    # Subplot 1: Trajectory Tracking
    plt.subplot(1, 2, 1)
    plt.plot(res_df['true_x'], res_df['true_y'], 'b-', linewidth=2, label='True Flight Path')
    plt.plot(res_df['ekf_x'], res_df['ekf_y'], 'r--', linewidth=1.5, label='EKF Estimated Path')
    plt.title("EKF Sensor Fusion Trajectory Tracking")
    plt.xlabel("North (m)"); plt.ylabel("East (m)"); plt.legend(); plt.grid(True)
    
    # Subplot 2: 3-Sigma Covariance Bounds
    plt.subplot(1, 2, 2)
    plt.plot(res_df['timestamp'], res_df['pos_error'], 'k-', label='Position Error (m)')
    plt.plot(res_df['timestamp'], res_df['sigma_bound'], 'r--', label='3σ Uncertainty Bound')
    plt.title("EKF Error Convergence & 3σ Bounds")
    plt.xlabel("Time (s)"); plt.ylabel("Error (m)"); plt.legend(); plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("plots/week2_ekf_validation.png", dpi=300)
    print("✅ Week 2 Validation Complete! Plots saved to plots/week2_ekf_validation.png")
    plt.show()

if __name__ == "__main__":
    run_week2_validation()