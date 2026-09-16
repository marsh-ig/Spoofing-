import os
import sys

# Guarantee Python finds root directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import matplotlib.pyplot as plt
from src.sim.trajectory_generator import generate_flight

def main():
    print("[*] Running UAV flight simulation...")
    df = generate_flight(duration_sec=60.0, dt=0.01)
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    df.to_csv("data/clean_flight.csv", index=False)
    print(f"[+] Saved {len(df)} telemetry samples to data/clean_flight.csv")
    
    # Plot
    fig = plt.figure(figsize=(11, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot(df['pos_x'], df['pos_y'], df['pos_z'], label='True Flight Path', color='blue', linewidth=2)
    gps_data = df.dropna(subset=['gps_x'])
    ax1.scatter(gps_data['gps_x'], gps_data['gps_y'], gps_data['gps_z'], color='red', s=4, alpha=0.6, label='10 Hz GPS Fixes')
    ax1.set_title("Clean UAV 3D Flight Trajectory")
    ax1.set_xlabel("North (m)"); ax1.set_ylabel("East (m)"); ax1.set_zlabel("Altitude (m)")
    ax1.legend()
    
    ax2 = fig.add_subplot(122)
    ax2.plot(df['timestamp'], df['imu_ax'], label='IMU Ax', color='green', alpha=0.6)
    ax2.plot(df['timestamp'], df['imu_ay'], label='IMU Ay', color='orange', alpha=0.6)
    ax2.set_title("100 Hz Synthetic IMU Noise")
    ax2.set_xlabel("Time (s)"); ax2.set_ylabel("Accel (m/s²)"); ax2.grid(True); ax2.legend()
    
    plt.tight_layout()
    plt.savefig("plots/clean_flight.png", dpi=300)
    print("✅ Plot saved to plots/clean_flight.png")
    plt.show()

if __name__ == "__main__":
    main()