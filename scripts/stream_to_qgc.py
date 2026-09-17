"""
Live QGroundControl (QGC) Telemetry Streaming Script
Simulates full 3D UAV flight, Slam-Frog GPS Spoofing Attack, and Layer 3 Active Probing Defense
streaming directly to QGroundControl in real-time over MAVLink UDP (port 14550).
"""
import os
import sys
import time
import math
import numpy as np
from pymavlink import mavutil

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sim.qgc_mavlink_bridge import QGCMavlinkBridge
from src.sim.trajectory_generator import generate_reference_trajectory
from src.sensors.imu_model import SyntheticIMU
from src.sensors.gps_model import SyntheticGPS
from src.filters.ekf_fusion import LooselyCoupledEKF
from src.attacks.slam_frog_spoofer import SlamFrogSpoofer
from src.control.maneuver_injector import MicroManeuverInjector
from src.detection.layer1_passive import Layer1PassiveDetector
from src.detection.layer3_active import Layer3ActiveTrap

def main():
    print("=" * 70)
    print("🛰️  UAV GPS INTEGRITY MONITORING — LIVE QGROUNDCONTROL (QGC) BRIDGE")
    print("=" * 70)
    print("[*] Target Ground Control Station: QGroundControl (127.0.0.1:14550)")
    print("[*] Launch QGroundControl on your laptop to see the live 3D map & HUD!")
    print("[*] Press Ctrl+C in this terminal anytime to stop simulation.\n")
    
    # 1. Initialize Simulation Physics & Telemetry Bridge
    dt_sim = 0.02  # 50 Hz physics update
    sim_duration = 60.0
    bridge = QGCMavlinkBridge(target_ip="127.0.0.1", target_port=14550, origin_lat=12.9716, origin_lon=77.5946, origin_alt=120.0)
    
    t_axis, true_states = generate_reference_trajectory(duration=sim_duration, dt=dt_sim)
    imu = SyntheticIMU(dt=dt_sim, seed=42)
    gps = SyntheticGPS(dt=0.10, seed=42)
    spoofer = SlamFrogSpoofer(attack_start_time=20.0, drift_accel=0.08)
    injector = MicroManeuverInjector(probe_times=[35.0], duration=0.3, amplitude_deg=1.0)
    ekf = LooselyCoupledEKF(dt_imu=dt_sim)
    layer1 = Layer1PassiveDetector(chi2_threshold=22.46)
    trap = Layer3ActiveTrap(min_correlation=0.70, max_lag_sec=0.08)
    
    # State tracking
    is_spoofing = False
    is_probing = False
    is_attack_detected = False
    is_gps_rejected = False
    
    # Buffers for Active Trap window evaluation
    imu_accel_buf = []
    gps_accel_buf = []
    
    print("[+] Sending initial MAVLink Heartbeat and Mission Start...")
    bridge.send_heartbeat()
    bridge.send_statustext("AVIONICS: Autonomous Mission Armed. EKF Online.", mavutil.mavlink.MAV_SEVERITY_NOTICE)
    
    start_wall_time = time.time()
    last_gps_time = 0.0
    last_hud_print = 0.0
    
    try:
        for idx, (t, state) in enumerate(zip(t_axis, true_states)):
            loop_start = time.time()
            
            # 1. IMU Physics & Measurement
            imu_meas = imu.measure(state)
            
            # Check for active probe injection at t=35s
            probe_cmd = injector.get_probe_command(t)
            if abs(probe_cmd[0]) > 0.001 and not is_probing:
                is_probing = True
                bridge.send_statustext("ACTION: Injecting 1 deg Active Micro-Maneuver Probe", mavutil.mavlink.MAV_SEVERITY_WARNING)
            
            # IMU body acceleration (aerodynamic response included)
            a_body = imu_meas.accel[1] + probe_cmd[0] * 65.0
            imu_accel_buf.append(a_body)
            
            # EKF Predict step (100 Hz / 50 Hz)
            ekf.predict(imu_meas.accel, imu_meas.gyro)
            
            # 2. GPS Measurement (10 Hz)
            if t - last_gps_time >= 0.10:
                last_gps_time = t
                raw_gps = gps.measure(state)
                
                # Apply attack after t=20s
                if t >= 20.0 and not is_spoofing:
                    is_spoofing = True
                    bridge.send_statustext("WARN: GPS Spoofing Drift Initiated (Slam-Frog sub-noise)", mavutil.mavlink.MAV_SEVERITY_WARNING)
                    
                spoofed_gps = spoofer.modify_gps(raw_gps, t)
                
                # Active GPS Acceleration response
                if is_spoofing and not is_gps_rejected:
                    # Non-reactive spoofer misses twitch
                    gps_resp_accel = np.random.normal(0, 0.05)
                else:
                    gps_resp_accel = a_body + np.random.normal(0, 0.05)
                gps_accel_buf.append(gps_resp_accel)
                
                # EKF Update step (if GPS not yet isolated by supervisor)
                if not is_gps_rejected:
                    gamma, S = ekf.update_gps(spoofed_gps.pos, spoofed_gps.vel)
                    d_M2, _ = layer1.evaluate(gamma, S)
                
                # Active Probing Evaluation after twitch at t=36.0s
                if t >= 36.0 and not is_attack_detected and is_probing:
                    res = trap.evaluate_window(np.array(imu_accel_buf), np.array(gps_accel_buf), dt=0.02)
                    peak_corr, lag, status = res[0], res[1], res[2]
                    
                    if "COMPROMISED" in status:
                        is_attack_detected = True
                        is_gps_rejected = True
                        bridge.send_statustext(f"CRITICAL: GPS Spoofing Confirmed! (Peak={peak_corr:.2f}, Lag={lag*1000:.0f}ms)", mavutil.mavlink.MAV_SEVERITY_CRITICAL)
                        bridge.send_statustext("EMERGENCY: GPS Rejected. Switched to IMU Dead-Reckoning.", mavutil.mavlink.MAV_SEVERITY_EMERGENCY)
                        print(f"\n🚨 [LAYER 3 ALARM TRIGGERED] {status} -> FAILOVER TO DEAD-RECKONING!\n")
            
            # 3. Stream Telemetry to QGroundControl
            bridge.send_heartbeat()
            
            # Current estimated position (EKF) vs GPS position
            ekf_state = ekf.get_state()
            x_est = ekf_state[0]
            y_est = ekf_state[1]
            z_est = state.pos[2]
            
            # Send Global Position, Attitude & HUD to QGC
            bridge.send_global_position(x_est, y_est, z_est, state.vel[0], state.vel[1], state.vel[2], state.yaw)
            bridge.send_attitude(state.roll, state.pitch, state.yaw)
            bridge.send_vfr_hud(
                airspeed=float(np.linalg.norm(state.vel)),
                groundspeed=float(np.linalg.norm(state.vel[:2])),
                heading_deg=float(math.degrees(state.yaw)),
                altitude=float(bridge.origin_alt + z_est),
                climb_rate=float(state.vel[2])
            )
            
            # 4. Terminal Progress HUD
            if t - last_hud_print >= 1.0:
                last_hud_print = t
                status_str = "HEALTHY" if not is_spoofing else ("SPOOFED (Layer 1 Blind)" if not is_gps_rejected else "PROTECTED (IMU Dead-Reckoning)")
                drift_m = np.linalg.norm(spoofed_gps.pos[:2] - state.pos[:2]) if is_spoofing else 0.0
                print(f"⏱️ T={t:4.1f}s | Pos: ({x_est:6.1f}m, {y_est:6.1f}m) | Speed: {np.linalg.norm(state.vel):4.1f} m/s | Spoof Drift: {drift_m:4.1f}m | Status: [{status_str}]")
            
            # Synchronize to real-time (1x real-time pacing)
            elapsed = time.time() - loop_start
            sleep_time = dt_sim - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
                
    except KeyboardInterrupt:
        print("\n[!] Simulation stopped by user.")
    
    print("\n[+] Simulation run complete! Telemetry successfully streamed to QGroundControl.")

if __name__ == "__main__":
    main()
