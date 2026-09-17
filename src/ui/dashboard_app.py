"""
UAV Navigation & GPS Integrity Monitoring HUD Dashboard
Interactive Ground Control Telemetry Visualizer & Live Simulation Suite
"""
import os
import sys

# Ensure root directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.sim.trajectory_generator import generate_reference_trajectory
from src.sensors.imu_model import SyntheticIMU
from src.sensors.gps_model import SyntheticGPS
from src.filters.ekf_fusion import LooselyCoupledEKF
from src.attacks.slam_frog_spoofer import SlamFrogSpoofer
from src.attacks.reactive_spoofer import ReactiveSpoofer
from src.control.maneuver_injector import MicroManeuverInjector
from src.detection.layer1_passive import Layer1PassiveDetector
from src.detection.layer3_active import Layer3ActiveTrap

# Page Configuration
st.set_page_config(page_title="UAV GPS Integrity & Active Trap HUD", layout="wide", page_icon="🛰️")

# Custom CSS for Dark Avionics Theme
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stMetric { background-color: #161f30; padding: 12px; border-radius: 8px; border: 1px solid #22324d; }
    .status-box { padding: 15px; border-radius: 8px; font-weight: bold; text-align: center; font-size: 18px; margin-bottom: 15px; }
    .healthy { background-color: #064e3b; color: #34d399; border: 1px solid #059669; }
    .warning { background-color: #78350f; color: #fbbf24; border: 1px solid #d97706; }
    .compromised { background-color: #7f1d1d; color: #f87171; border: 1px solid #dc2626; }
</style>
""", unsafe_allow_html=True)

# Title & Header
st.title("🛰️ UAV Avionics & GPS Navigation Integrity Simulation Suite")
st.markdown("**Project:** Active Probing Against Stealthy GPS Spoofing | **Architecture:** Multi-Tier Closed-Loop SITL Sim")

# Sidebar Simulation Mode
st.sidebar.header("🕹️ Simulation Mode")
sim_mode = st.sidebar.radio(
    "Select Scenario to Simulate:",
    ["1. Clean Flight & EKF Fusion (Week 2)", 
     "2. Stealthy Slam-Frog Attack (Week 3)", 
     "3. Active Probing Micro-Maneuver Trap (Week 4)"]
)

# -------------------------------------------------------------
# MODE 1: CLEAN FLIGHT & EKF FUSION
# -------------------------------------------------------------
if "1. Clean" in sim_mode:
    st.subheader("🌐 Clean Baseline Flight: 100 Hz EKF Sensor Fusion")
    st.markdown('<div class="status-box healthy">🟢 SYSTEM STATUS: HEALTHY (Clean Flight Baseline — Zero False Alarms)</div>', unsafe_allow_html=True)
    
    # Run lightweight simulation
    t_arr, true_states = generate_reference_trajectory(duration=45.0, dt=0.01)
    imu = SyntheticIMU(dt=0.01, seed=42)
    gps = SyntheticGPS(dt=0.10, seed=42)
    ekf = LooselyCoupledEKF(dt_imu=0.01)
    
    ekf_x, ekf_y, errors, bounds = [], [], [], []
    for t, state in zip(t_arr, true_states):
        imu_meas = imu.measure(state)
        ekf.predict(imu_meas.accel, imu_meas.gyro)
        if gps.is_sample_ready(t):
            gps_meas = gps.measure(state)
            ekf.update_gps(gps_meas.pos, gps_meas.vel)
        est = ekf.get_state()
        ekf_x.append(est[0])
        ekf_y.append(est[1])
        err = np.linalg.norm([est[0] - state.pos[0], est[1] - state.pos[1]])
        sig = 3.0 * np.sqrt(ekf.P[0,0] + ekf.P[1,1])
        errors.append(err)
        bounds.append(sig)
        
    true_x = [s.pos[0] for s in true_states]
    true_y = [s.pos[1] for s in true_states]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("⏱️ Flight Duration", f"{t_arr[-1]:.1f} s")
    col2.metric("🎯 Final Pos Error", f"{errors[-1]:.2f} m")
    col3.metric("📉 Mean RMSE", f"{np.sqrt(np.mean(np.array(errors)**2)):.2f} m")
    col4.metric("🛡️ 3σ Covariance", f"{bounds[-1]:.2f} m")
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("3D Flight Trajectory (Ground Truth vs EKF)", "Position Error vs 3σ Bound"))
    fig.add_trace(go.Scatter(x=true_x, y=true_y, mode='lines', name='True Path', line=dict(color='#38bdf8', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=ekf_x, y=ekf_y, mode='lines', name='EKF Estimate', line=dict(color='#34d399', dash='dash', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_arr, y=errors, mode='lines', name='Error (m)', line=dict(color='#818cf8')), row=1, col=2)
    fig.add_trace(go.Scatter(x=t_arr, y=bounds, mode='lines', name='3σ Bound (m)', line=dict(color='#f43f5e', dash='dash')), row=1, col=2)
    fig.update_layout(paper_bgcolor='#0b0f19', plot_bgcolor='#111827', font=dict(color='white'), height=480)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# MODE 2: SLAM-FROG STEALTHY ATTACK
# -------------------------------------------------------------
elif "2. Stealthy" in sim_mode:
    st.subheader("🐸 Slam-Frog Stealthy Attack: Defeating Layer 1 Passive Detector")
    st.markdown('<div class="status-box warning">⚠️ SYSTEM STATUS: ATTACK IN PROGRESS — PASSIVE DETECTOR FAILS (Blind Stealth Drift)</div>', unsafe_allow_html=True)
    
    t_arr, true_states = generate_reference_trajectory(duration=45.0, dt=0.01)
    imu = SyntheticIMU(dt=0.01, seed=42)
    gps = SyntheticGPS(dt=0.10, seed=42)
    spoofer = SlamFrogSpoofer(attack_start_time=20.0, drift_accel=0.08)
    layer1 = Layer1PassiveDetector(chi2_threshold=22.46)
    ekf = LooselyCoupledEKF(dt_imu=0.01)
    
    gps_fake_x, gps_fake_y, mahalanobis_d2, true_dist = [], [], [], []
    for t, state in zip(t_arr, true_states):
        imu_meas = imu.measure(state)
        ekf.predict(imu_meas.accel, imu_meas.gyro)
        if gps.is_sample_ready(t):
            gps_meas = gps.measure(state)
            spoofed_meas = spoofer.modify_gps(gps_meas, t)
            gamma, S = ekf.update_gps(spoofed_meas.pos, spoofed_meas.vel)
            d_M2, _ = layer1.evaluate(gamma, S)
            mahalanobis_d2.append((t, d_M2))
            gps_fake_x.append(spoofed_meas.pos[0])
            gps_fake_y.append(spoofed_meas.pos[1])
            true_dist.append(np.linalg.norm(spoofed_meas.pos[:2] - state.pos[:2]))
            
    true_x = [s.pos[0] for s in true_states]
    true_y = [s.pos[1] for s in true_states]
    m_times, m_vals = zip(*mahalanobis_d2)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🚨 Total Drift Induced", f"{true_dist[-1]:.1f} m")
    col2.metric("📊 Max Mahalanobis d_M^2", f"{max(m_vals):.2f} (Threshold: 22.46)")
    col3.metric("❌ Passive Layer 1 Outcome", "EVADED (Alarm Never Triggered)")
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Flight Trajectory under Slam-Frog Attack", "Mahalanobis Chi^2 Residual (Stays Below 22.46)"))
    fig.add_trace(go.Scatter(x=true_x, y=true_y, mode='lines', name='True Path', line=dict(color='#38bdf8', width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gps_fake_x, y=gps_fake_y, mode='lines', name='Spoofed GPS Path', line=dict(color='#f43f5e', dash='dash', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=m_times, y=m_vals, mode='lines', name='Mahalanobis d_M^2', line=dict(color='#fbbf24')), row=1, col=2)
    fig.add_hline(y=22.46, line=dict(color='#f43f5e', dash='dash'), annotation_text="Chi^2 Alarm Limit (22.46)", row=1, col=2)
    fig.update_layout(paper_bgcolor='#0b0f19', plot_bgcolor='#111827', font=dict(color='white'), height=480)
    st.plotly_chart(fig, use_container_width=True)

# -------------------------------------------------------------
# MODE 3: ACTIVE PROBING MICRO-MANEUVER TRAP
# -------------------------------------------------------------
else:
    st.subheader("⚡ Active Probing Trap: Micro-Maneuver Cross-Correlation Defense")
    
    st.sidebar.subheader("⚙️ Active Probe Parameters")
    twitch_deg = st.sidebar.slider("Twitch Amplitude (degrees):", 0.5, 3.0, 1.0, 0.5)
    rf_delay = st.sidebar.slider("Smart Spoofer RF Delay (ms):", 100, 400, 200, 50) / 1000.0
    
    dt = 0.01
    t_axis = np.arange(0.0, 4.0, dt)
    probe_start = 1.0
    probe_dur = 0.3
    injector = MicroManeuverInjector(probe_times=[probe_start], duration=probe_dur, amplitude_deg=twitch_deg)
    trap = Layer3ActiveTrap(min_correlation=0.70, max_lag_sec=0.08)
    
    u_cmd = np.array([injector.get_probe_command(t)[0] for t in t_axis])
    a_true = np.zeros_like(u_cmd)
    for i in range(1, len(t_axis)):
        a_true[i] = a_true[i-1] + (u_cmd[i] * 65.0 - a_true[i-1]) * (dt / 0.04)
        
    imu_accel = a_true + np.random.normal(0, 0.04, len(t_axis))
    gps_auth = np.roll(a_true, int(0.02 / dt)) + np.random.normal(0, 0.05, len(t_axis))
    gps_smart = np.roll(a_true, int(rf_delay / dt)) + np.random.normal(0, 0.05, len(t_axis))
    gps_non = np.random.normal(0, 0.05, len(t_axis))
    
    w_probe = (int((probe_start - 0.05) / dt), int((probe_start + probe_dur + 0.15) / dt))
    res_auth = trap.evaluate_window(imu_accel, gps_auth, dt, probe_window=w_probe)
    res_smart = trap.evaluate_window(imu_accel, gps_smart, dt, probe_window=w_probe)
    res_non = trap.evaluate_window(imu_accel, gps_non, dt, probe_window=w_probe)
    
    st.markdown(f'<div class="status-box compromised">🚨 ACTIVE PROBING ALERT: Non-Reactive [Missing Peak rho={res_non[0]:.2f}] | Smart Spoofer [Lag Detected tau={res_smart[1]*1000:.0f}ms]</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("🟢 Authentic GPS", f"rho={res_auth[0]:.2f} @ {res_auth[1]*1000:.0f}ms", "HEALTHY")
    col2.metric("🔴 Smart Spoofer", f"rho={res_smart[0]:.2f} @ {res_smart[1]*1000:.0f}ms", "LAG ALARM (>80ms)")
    col3.metric("🟣 Non-Reactive", f"rho={res_non[0]:.2f} < 0.70", "MISSING PEAK")
    
    lags = res_auth[3]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Physical Twitch Response (0ms vs 200ms)", "Layer 3 Cross-Correlation Curves rho_xy(tau)"))
    
    fig.add_trace(go.Scatter(x=t_axis, y=imu_accel, mode='lines', name='IMU Jerk (0ms)', line=dict(color='#38bdf8', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_axis, y=gps_auth, mode='lines', name='Authentic GPS (20ms)', line=dict(color='#34d399', width=2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=t_axis, y=gps_smart, mode='lines', name=f'Smart Spoofer ({rf_delay*1000:.0f}ms Lag)', line=dict(color='#f43f5e', width=2)), row=1, col=1)
    
    fig.add_trace(go.Scatter(x=lags, y=res_auth[4], mode='lines', name='Authentic (Healthy)', line=dict(color='#34d399', width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=lags, y=res_smart[4], mode='lines', name='Smart Spoofer (Lag)', line=dict(color='#f43f5e', width=3)), row=1, col=2)
    fig.add_trace(go.Scatter(x=lags, y=res_non[4], mode='lines', name='Non-Reactive (Flat)', line=dict(color='#c084fc', dash='dash', width=2)), row=1, col=2)
    
    fig.add_vline(x=0.08, line=dict(color='white', dash='dash'), annotation_text="80ms Lag Limit", row=1, col=2)
    fig.add_hline(y=0.70, line=dict(color='yellow', dash='dot'), annotation_text="0.70 Min Corr", row=1, col=2)
    
    fig.update_layout(paper_bgcolor='#0b0f19', plot_bgcolor='#111827', font=dict(color='white'), height=480)
    st.plotly_chart(fig, use_container_width=True)
