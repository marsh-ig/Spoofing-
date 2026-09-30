"""
MAVLink Man-in-the-Middle (MITM) Security Proxy.
Enables single-machine GPS spoofing attack injection and Active Probing Defense
by intercepting and manipulating MAVLink telemetry streams in memory between PX4 and QGC.
"""
import time
import math
import numpy as np
from pymavlink import mavutil
from src.sensors.gps_almanac_engine import GPSAlmanacEngine
from src.attacks.slam_frog_spoofer import GPSSpoofer
from src.control.maneuver_injector import MicroManeuverInjector
from src.detection.layer3_active import Layer3ActiveTrap

class MAVLinkSecurityProxy:
    def __init__(self, upstream_port: int = 14540, downstream_port: int = 14550, origin_lat: float = 12.9716, origin_lon: float = 77.5946):
        """
        Initializes the in-memory MAVLink Proxy on a single machine.
        - upstream: connects to PX4 / SITL Generator (port 14540)
        - downstream: streams modified telemetry to QGroundControl (port 14550)
        """
        self.upstream_port = upstream_port
        self.downstream_port = downstream_port
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        
        # Initialize Almanac Engine & Security Modules
        self.almanac = GPSAlmanacEngine(origin_lat=origin_lat, origin_lon=origin_lon)
        self.spoofer = GPSSpoofer(attack_type="slam_frog", start_time=20.0)
        self.injector = MicroManeuverInjector(probe_times=[35.0], duration=0.3, amplitude_deg=1.0)
        self.trap = Layer3ActiveTrap(min_correlation=0.70, max_lag_sec=0.08)
        
        # MAVLink Downstream to QGC
        self.qgc_link = mavutil.mavlink_connection(f"udpout:127.0.0.1:{downstream_port}", source_system=1, source_component=1)
        self.boot_time = time.time()
        
        # Internal State
        self.is_spoofing = False
        self.is_probing = False
        self.is_attack_detected = False
        self.is_gps_rejected = False
        
        self.imu_accel_history = []
        self.gps_accel_history = []

    def boot_ms(self) -> int:
        return int((time.time() - self.boot_time) * 1000)

    def process_and_forward_telemetry(self, t_sec: float, uav_pos_ned: np.ndarray, uav_vel_ned: np.ndarray, uav_accel_ned: np.ndarray, roll: float, pitch: float, yaw: float):
        """
        Intercepts UAV state, calculates orbital GPS pseudoranges, injects stealthy spoofing drift,
        applies active probing cross-correlation defense, and forwards MAVLink packets to QGC.
        """
        # 1. Compute Realistic Orbital GPS Observations
        sats_obs, dop = self.almanac.observe_constellation(uav_pos_ned, uav_vel_ned, t_sec)
        
        # 2. Check for Active Micro-Maneuver Twitch Probe
        probe_cmd = self.injector.get_probe_command(t_sec)
        if abs(probe_cmd[0]) > 0.001 and not self.is_probing:
            self.is_probing = True
            self.send_statustext("ACTION: Injecting 1 deg Active Micro-Maneuver Probe", mavutil.mavlink.MAV_SEVERITY_WARNING)
            
        # Aerodynamic acceleration signature
        a_twitch = probe_cmd[0] * 65.0
        imu_measured_accel = uav_accel_ned[1] + a_twitch + np.random.normal(0, 0.04)
        self.imu_accel_history.append(imu_measured_accel)
        
        # 3. Apply Attack (if t >= 20s)
        reported_pos = uav_pos_ned.copy()
        reported_vel = uav_vel_ned.copy()
        
        if t_sec >= 20.0 and not self.is_spoofing:
            self.is_spoofing = True
            self.send_statustext("WARN: Sub-noise Slam-Frog GPS drift initiated", mavutil.mavlink.MAV_SEVERITY_WARNING)
            
        if self.is_spoofing and not self.is_gps_rejected:
            dt_atk = t_sec - 20.0
            drift_accel = self.spoofer.drift_accel
            reported_pos += 0.5 * drift_accel * (dt_atk ** 2)
            reported_vel += drift_accel * dt_atk
            # Non-reactive spoofer misses twitch
            gps_measured_accel = np.random.normal(0, 0.05)
        else:
            gps_measured_accel = imu_measured_accel + np.random.normal(0, 0.05)
            
        self.gps_accel_history.append(gps_measured_accel)
        
        # 4. Layer 3 Active Probing Trap Check
        if t_sec >= 36.0 and not self.is_attack_detected and self.is_probing:
            res = self.trap.evaluate_window(np.array(self.imu_accel_history), np.array(self.gps_accel_history), dt=0.02)
            peak_corr, lag, status = res[0], res[1], res[2]
            
            if "COMPROMISED" in status:
                self.is_attack_detected = True
                self.is_gps_rejected = True
                self.send_statustext(f"CRITICAL: GPS Spoofing Confirmed! (Peak={peak_corr:.2f}, Lag={lag*1000:.0f}ms)", mavutil.mavlink.MAV_SEVERITY_CRITICAL)
                self.send_statustext("EMERGENCY: GPS Rejected. Switched to IMU Dead-Reckoning.", mavutil.mavlink.MAV_SEVERITY_EMERGENCY)
                print(f"\n🚨 [PROXY LAYER 3 TRAP TRIGGERED] {status} -> FAILOVER TO DEAD-RECKONING!\n")

        # 5. If GPS is isolated, fallback to Dead-Reckoning
        display_pos = uav_pos_ned if self.is_gps_rejected else reported_pos
        display_vel = uav_vel_ned if self.is_gps_rejected else reported_vel
        
        # 6. Transmit MAVLink Packets to QGroundControl
        self.send_heartbeat_and_sys_status()
        self.send_telemetry(display_pos, display_vel, roll, pitch, yaw, dop["HDOP"], dop["VDOP"], dop["SatsVisible"])

    def send_heartbeat_and_sys_status(self):
        """Sends MAVLink Heartbeat and SYS_STATUS."""
        # Heartbeat
        self.qgc_link.mav.heartbeat_send(
            mavutil.mavlink.MAV_TYPE_QUADROTOR,
            mavutil.mavlink.MAV_AUTOPILOT_GENERIC,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
            0,
            mavutil.mavlink.MAV_STATE_ACTIVE
        )
        # System status
        sensors = (
            mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO |
            mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_ACCEL |
            mavutil.mavlink.MAV_SYS_STATUS_SENSOR_GPS
        )
        self.qgc_link.mav.sys_status_send(
            sensors, sensors, sensors,
            500, 12400, -1, 85, 0, 0, 0, 0, 0, 0
        )
        # Auto-responder dummy param
        param_id = b'SYS_AUTOSTART\x00\x00\x00'
        self.qgc_link.mav.param_value_send(param_id, 4001.0, mavutil.mavlink.MAV_PARAM_TYPE_REAL32, 1, 0)

    def send_telemetry(self, pos_ned: np.ndarray, vel_ned: np.ndarray, roll: float, pitch: float, yaw: float, hdop: float, vdop: float, num_sats: int):
        """Sends Global Position, Attitude, Raw GPS and HUD instruments to QGC."""
        lat_deg = self.origin_lat + (pos_ned[0] / 111139.0)
        lon_deg = self.origin_lon + (pos_ned[1] / (111139.0 * math.cos(math.radians(self.origin_lat))))
        alt_amsl_mm = int((self.almanac.origin_alt - pos_ned[2]) * 1000)
        hdg_cdeg = int((math.degrees(yaw) % 360) * 100)
        speed_ms = float(np.linalg.norm(vel_ned[:2]))
        
        # GLOBAL_POSITION_INT
        self.qgc_link.mav.global_position_int_send(
            self.boot_ms(),
            int(lat_deg * 1e7),
            int(lon_deg * 1e7),
            alt_amsl_mm,
            int(-pos_ned[2] * 1000),
            int(vel_ned[0] * 100),
            int(vel_ned[1] * 100),
            int(vel_ned[2] * 100),
            hdg_cdeg
        )
        
        # GPS_RAW_INT (with real almanac HDOP / VDOP / Sat count)
        self.qgc_link.mav.gps_raw_int_send(
            int(time.time() * 1e6),
            3,
            int(lat_deg * 1e7),
            int(lon_deg * 1e7),
            alt_amsl_mm,
            int(hdop * 100),
            int(vdop * 100),
            int(speed_ms * 100),
            hdg_cdeg,
            num_sats
        )
        
        # ATTITUDE
        self.qgc_link.mav.attitude_send(self.boot_ms(), roll, pitch, yaw, 0.0, 0.0, 0.0)
        
        # VFR_HUD
        self.qgc_link.mav.vfr_hud_send(
            float(np.linalg.norm(vel_ned)),
            speed_ms,
            int(math.degrees(yaw) % 360),
            75,
            float(self.almanac.origin_alt - pos_ned[2]),
            float(-vel_ned[2])
        )

    def send_statustext(self, message: str, severity: int = mavutil.mavlink.MAV_SEVERITY_INFO):
        text_bytes = message.encode('utf-8')[:50]
        self.qgc_link.mav.statustext_send(severity, text_bytes)
        print(f"[MITM PROXY STATUSTEXT] {message}")
