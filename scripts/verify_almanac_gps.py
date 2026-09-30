"""
Verification & Validation of Keplerian GPS Almanac Engine & Physical Signal Variance.
Demonstrates realistic satellite visibility, GDOP/PDOP variance, pseudorange generation,
and the geometric projection of Active Micro-Maneuver Probing across the constellation.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sim.trajectory_generator import generate_flight
from src.sensors.gps_almanac_engine import GPSAlmanacEngine
from src.control.maneuver_injector import MicroManeuverInjector
from src.detection.layer3_active import Layer3ActiveTrap

def main():
    print("=" * 70)
    print("[*] KEPLERIAN GPS ALMANAC & CONSTELLATION DYNAMICS VALIDATION")
    print("=" * 70)
    
    dt = 0.02
    duration = 60.0
    df_flight = generate_flight(duration_sec=duration, dt=dt)
    
    almanac = GPSAlmanacEngine(origin_lat=12.9716, origin_lon=77.5946, origin_alt=100.0)
    injector = MicroManeuverInjector(probe_times=[35.0], duration=0.3, amplitude_deg=1.0)
    trap = Layer3ActiveTrap(min_correlation=0.70, max_lag_sec=0.08)
    
    time_list = []
    gdop_list = []
    pdop_list = []
    hdop_list = []
    sats_vis_list = []
    pr_samples = {prn: [] for prn in [1, 4, 7, 10, 13, 16]}
    
    imu_accel_proj = []
    gps_auth_proj = []
    gps_smart_proj = []
    gps_non_proj = []
    
    print("[*] Propagating 31-Satellite GPS Constellation across flight trajectory...")
    
    for idx, row in df_flight.iterrows():
        t = row['timestamp']
        pos_ned = np.array([row['pos_x'], row['pos_y'], -row['pos_z']])
        vel_ned = np.array([row['vel_x'], row['vel_y'], row['vel_z']])
        
        # Micro-maneuver command
        u_cmd = injector.get_probe_command(t)[0]
        a_twitch = u_cmd * 65.0 # Lateral acceleration
        
        obs, dop = almanac.observe_constellation(pos_ned, vel_ned, t)
        
        time_list.append(t)
        gdop_list.append(dop["GDOP"])
        pdop_list.append(dop["PDOP"])
        hdop_list.append(dop["HDOP"])
        sats_vis_list.append(dop["SatsVisible"])
        
        # Track selected PRN pseudoranges
        obs_dict = {o.prn: o for o in obs}
        for prn in pr_samples:
            if prn in obs_dict:
                pr_samples[prn].append(obs_dict[prn].pseudorange / 1000.0)
            else:
                pr_samples[prn].append(np.nan)
                
        # Constellation geometric projection of twitch onto high-elevation satellite (PRN 07)
        if 7 in obs_dict:
            los = obs_dict[7].los_unit_vector
            # Projection of twitch lateral body axis (East/y-axis) onto satellite LOS
            proj_weight = float(los[1])
            
            # IMU measures true jerk instantly
            imu_accel_proj.append(a_twitch * proj_weight + np.random.normal(0, 0.02))
            
            # Authentic GPS: 20ms tracking loop delay
            gps_auth_proj.append(a_twitch * proj_weight + np.random.normal(0, 0.03))
            
            # Smart Spoofer: 200ms RF lag
            gps_smart_proj.append(a_twitch * proj_weight + np.random.normal(0, 0.03))
            
            # Non-reactive spoofer: zero twitch projection
            gps_non_proj.append(np.random.normal(0, 0.03))
        else:
            imu_accel_proj.append(0.0)
            gps_auth_proj.append(0.0)
            gps_smart_proj.append(0.0)
            gps_non_proj.append(0.0)

    # Shift for delays
    gps_auth_proj = np.roll(np.array(gps_auth_proj), int(0.02 / dt))
    gps_smart_proj = np.roll(np.array(gps_smart_proj), int(0.20 / dt))
    
    # Active probing evaluation on constellation projection
    probe_idx = int(35.0 / dt)
    w_eval = (probe_idx - int(0.05/dt), probe_idx + int(0.45/dt))
    res_auth = trap.evaluate_window(np.array(imu_accel_proj), gps_auth_proj, dt, probe_window=w_eval)
    res_smart = trap.evaluate_window(np.array(imu_accel_proj), gps_smart_proj, dt, probe_window=w_eval)
    res_non = trap.evaluate_window(np.array(imu_accel_proj), np.array(gps_non_proj), dt, probe_window=w_eval)
    
    print(f"\n[+] GPS Constellation Metrics:")
    print(f" -> Mean Visible Satellites : {np.mean(sats_vis_list):.1f} satellites")
    print(f" -> Mean GDOP              : {np.mean(gdop_list):.2f}")
    print(f" -> Mean PDOP              : {np.mean(pdop_list):.2f}")
    print(f" -> Mean HDOP              : {np.mean(hdop_list):.2f}")
    
    print(f"\n[+] Active Micro-Maneuver Constellation Probing Results:")
    print(f" -> Authentic GPS          : Peak rho={res_auth[0]:.2f} @ tau={res_auth[1]*1000:3.0f}ms -> [{res_auth[2]}]")
    print(f" -> Smart Reactive Spoofer : Peak rho={res_smart[0]:.2f} @ tau={res_smart[1]*1000:3.0f}ms -> [{res_smart[2]}]")
    print(f" -> Non-Reactive Spoofer   : Peak rho={res_non[0]:.2f} @ tau={res_non[1]*1000:3.0f}ms -> [{res_non[2]}]")
    
    # Plotting Figure
    fig = plt.figure(figsize=(15, 10))
    
    # Subplot 1: Skyplot / Satellite Visibility
    ax1 = fig.add_subplot(221, projection='polar')
    obs_t0, _ = almanac.observe_constellation(np.zeros(3), np.zeros(3), t_sec=0.0)
    for sat in obs_t0:
        az_rad = np.radians(sat.azimuth_deg)
        r_zenith = 90.0 - sat.elevation_deg # Zenith angle (0 at center, 90 at horizon)
        ax1.plot(az_rad, r_zenith, 'o', markersize=10, label=f'PRN {sat.prn:02d}')
        ax1.text(az_rad, r_zenith + 3.0, f'{sat.prn:02d}', fontsize=8, fontweight='bold')
    ax1.set_theta_zero_location('N')
    ax1.set_theta_direction(-1) # Clockwise
    ax1.set_rlim(0, 90)
    ax1.set_rticks([15, 30, 45, 60, 75, 90])
    ax1.set_yticklabels(['75°', '60°', '45°', '30°', '15°', '0° (Horizon)'])
    ax1.set_title("(a) GPS Skyplot: Visible Satellite Constellation (10° Mask)", fontweight='bold', pad=15)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 2: Dilution of Precision (DOP)
    ax2 = fig.add_subplot(222)
    ax2.plot(time_list, gdop_list, 'b-', linewidth=2, label=f'GDOP (Mean: {np.mean(gdop_list):.2f})')
    ax2.plot(time_list, pdop_list, 'g--', linewidth=2, label=f'PDOP (Mean: {np.mean(pdop_list):.2f})')
    ax2.plot(time_list, hdop_list, 'm:', linewidth=2, label=f'HDOP (Mean: {np.mean(hdop_list):.2f})')
    ax2.set_title("(b) Dilution of Precision (DOP) over 60s Flight", fontweight='bold')
    ax2.set_xlabel("Flight Time (s)")
    ax2.set_ylabel("DOP Metric (Lower is Better)")
    ax2.set_ylim(0, 6)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 3: Pseudorange Evolution
    ax3 = fig.add_subplot(223)
    for prn in [1, 4, 7]:
        if not np.all(np.isnan(pr_samples[prn])):
            ax3.plot(time_list, pr_samples[prn], linewidth=2, label=f'PRN {prn:02d} Pseudorange')
    ax3.set_title("(c) Physical Pseudoranges with Ionospheric & Clock Drift", fontweight='bold')
    ax3.set_xlabel("Flight Time (s)")
    ax3.set_ylabel("Pseudorange rho_j (km)")
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    # Subplot 4: Active Probe Micro-Maneuver Traps on Constellation Projection
    ax4 = fig.add_subplot(224)
    lags = res_auth[3]
    ax4.axvspan(0.08, 0.40, color='#fee2e2', alpha=0.5, label='ALARM (Lag > 80ms)')
    ax4.axhspan(-0.5, 0.70, color='#fef3c7', alpha=0.4, label='ALARM (Peak < 0.70)')
    ax4.axvspan(-0.05, 0.08, color='#ecfdf5', alpha=0.4, label='HEALTHY ZONE')
    
    ax4.plot(lags, res_auth[4], 'g-', linewidth=2.5, label=f'Authentic: Peak={res_auth[0]:.2f} @ {res_auth[1]*1000:.0f}ms')
    ax4.plot(lags, res_smart[4], 'r-', linewidth=2.5, label=f'Smart Spoofer: Peak={res_smart[0]:.2f} @ {res_smart[1]*1000:.0f}ms (LAG)')
    ax4.plot(lags, res_non[4], 'm--', linewidth=2, label=f'Non-Reactive: Peak={res_non[0]:.2f} < 0.70 (MISSING)')
    
    ax4.axvline(x=0.08, color='black', linestyle='--', linewidth=1.5)
    ax4.axhline(y=0.70, color='black', linestyle=':', linewidth=1.5)
    ax4.set_title("(d) Constellation-Projected Active Probe Trap", fontweight='bold')
    ax4.set_xlabel("Estimated Delay tau (s)")
    ax4.set_ylabel("Cross-Correlation rho_xy(tau)")
    ax4.set_ylim(-0.4, 1.1)
    ax4.legend(loc='lower left', fontsize=8.5)
    ax4.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", "almanac_gps_validation.png")
    plt.savefig(out_path, dpi=300)
    print(f"\n[+] Validation plot successfully saved to {out_path}")
    if "--show" in sys.argv:
        plt.show()
    else:
        plt.close()

if __name__ == "__main__":
    main()
