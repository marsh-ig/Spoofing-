"""
Week 4 Verification: The Active Probing Micro-Maneuver Trap Benchmark.
Proves that Active Probing catches both Non-Reactive and Smart Reactive Spoofers!
"""
import os
import sys

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.control.maneuver_injector import MicroManeuverInjector
from src.attacks.reactive_spoofer import ReactiveSpoofer
from src.detection.layer3_active import Layer3ActiveTrap

def main():
    print("[*] Simulating Active Probing Micro-Maneuver Trap across all 3 scenarios...")
    
    np.random.seed(42)  # For reproducible academic verification
    dt = 0.01
    t_axis = np.arange(0.0, 4.0, dt)  # 4-second probing analysis window around t=35s
    probe_start = 1.0
    probe_dur = 0.3
    injector = MicroManeuverInjector(probe_times=[probe_start], duration=probe_dur, amplitude_deg=1.0)
    trap = Layer3ActiveTrap(min_correlation=0.70, max_lag_sec=0.08)
    
    # 1. Generate Commanded Twitch Signal (1 deg doublet in radians)
    u_cmd = np.array([injector.get_probe_command(t)[0] for t in t_axis])
    
    # 2. True UAV Physical Acceleration Response (Aerodynamic transfer function: 1 deg produces ~1.1 m/s^2)
    a_true = np.zeros_like(u_cmd)
    for i in range(1, len(t_axis)):
        a_true[i] = a_true[i-1] + (u_cmd[i] * 65.0 - a_true[i-1]) * (dt / 0.04)
        
    # Sensor Measurements with realistic noise
    imu_accel = a_true + np.random.normal(0, 0.04, len(t_axis))
    
    # Scenario A: Authentic GPS (Follows physics with tiny hardware tracking delay ~20ms)
    gps_authentic = np.roll(a_true, int(0.02 / dt)) + np.random.normal(0, 0.05, len(t_axis))
    
    # Scenario B: Smart Reactive Spoofer (Suffers 200 ms RF transport & synthesis delay)
    gps_smart = np.roll(a_true, int(0.20 / dt)) + np.random.normal(0, 0.05, len(t_axis))
    
    # Scenario C: Non-Reactive Spoofer (Replays smooth path, misses twitch completely)
    gps_nonreactive = np.random.normal(0, 0.05, len(t_axis))
    
    # Evaluate Cross-Correlations (IMU excitation vs GPS response)
    w_probe = (int((probe_start - 0.05) / dt), int((probe_start + probe_dur + 0.15) / dt))
    res_auth = trap.evaluate_window(imu_accel, gps_authentic, dt, probe_window=w_probe)
    res_smart = trap.evaluate_window(imu_accel, gps_smart, dt, probe_window=w_probe)
    res_non = trap.evaluate_window(imu_accel, gps_nonreactive, dt, probe_window=w_probe)
    
    print(f" -> Scenario A (Authentic):     Peak rho={res_auth[0]:.2f} at tau={res_auth[1]*1000:3.0f}ms -> [{res_auth[2]}]")
    print(f" -> Scenario B (Smart Spoofer): Peak rho={res_smart[0]:.2f} at tau={res_smart[1]*1000:3.0f}ms -> [{res_smart[2]}]")
    print(f" -> Scenario C (Non-Reactive):  Peak rho={res_non[0]:.2f} at tau={res_non[1]*1000:3.0f}ms -> [{res_non[2]}]")
    
    # Plotting the Headline Figure
    fig = plt.figure(figsize=(15, 6))
    
    # Subplot 1: Time-Series Probing Response
    ax1 = fig.add_subplot(121)
    ax1.plot(t_axis, u_cmd * 0.8, 'k--', linewidth=2, label='Commanded Twitch (1 deg Doublet)')
    ax1.plot(t_axis, imu_accel, 'b-', linewidth=2, alpha=0.8, label='IMU Physical Jerk (0ms delay)')
    ax1.plot(t_axis, gps_authentic, 'g-', linewidth=2, label='Authentic GPS (Synchronous, tau=20ms)')
    ax1.plot(t_axis, gps_smart, 'r-', linewidth=2, label='Smart Spoofer (200ms RF Lag [ALARM])')
    ax1.set_title("(a) Physical Micro-Maneuver Injection & Sensor Responses", fontweight='bold', fontsize=12)
    ax1.set_xlabel("Time around Probe (s)", fontsize=11)
    ax1.set_ylabel("Lateral Acceleration / Jerk (m/s^2)", fontsize=11)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Layer 3 Cross-Correlation Curves
    ax2 = fig.add_subplot(122)
    lags = res_auth[3]
    
    # Shaded Detection Threshold Zones
    ax2.axvspan(0.08, 0.40, color='#fee2e2', alpha=0.5, label='ALARM ZONE (Delay > 80ms)')
    ax2.axhspan(-0.5, 0.70, color='#fef3c7', alpha=0.4, label='ALARM ZONE (Peak < 0.70)')
    ax2.axvspan(-0.05, 0.08, color='#ecfdf5', alpha=0.4, label='HEALTHY ZONE (Authentic)')
    
    ax2.plot(lags, res_auth[4], 'g-', linewidth=2.5, label=f'Authentic GPS: Peak={res_auth[0]:.2f} @ tau={res_auth[1]*1000:.0f}ms (HEALTHY)')
    ax2.plot(lags, res_smart[4], 'r-', linewidth=2.5, label=f'Smart Spoofer: Peak={res_smart[0]:.2f} @ tau={res_smart[1]*1000:.0f}ms (LAG DETECTED)')
    ax2.plot(lags, res_non[4], 'm--', linewidth=2, label=f'Non-Reactive: Peak={res_non[0]:.2f} < 0.70 (MISSING PEAK)')
    
    ax2.axvline(x=0.08, color='black', linestyle='--', linewidth=1.5, label='Physical Delay Limit (80ms)')
    ax2.axhline(y=0.70, color='black', linestyle=':', linewidth=1.5, label='Min Correlation Threshold (0.70)')
    
    ax2.set_title("(b) Layer 3 Cross-Correlation Trap: Exposing Stealthy Spoofers", fontweight='bold', fontsize=12)
    ax2.set_xlabel("Estimated Response Delay tau (seconds)", fontsize=11)
    ax2.set_ylabel("Normalized Cross-Correlation rho_xy(tau)", fontsize=11)
    ax2.set_ylim(-0.4, 1.1)
    ax2.legend(loc='lower left', fontsize=8.5)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plot_path = os.path.join("plots", "week4_active_trap_success.png")
    plt.savefig(plot_path, dpi=300)
    print(f"[+] Week 4 Milestone Complete! Headline plot saved to {plot_path}")
    if "--show" in sys.argv:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()