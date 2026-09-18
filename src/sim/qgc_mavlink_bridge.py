"""
QGroundControl (QGC) MAVLink Telemetry Bridge
Streams real-time UAV flight simulation, GPS spoofing attacks, and Active Probing Defense
directly to QGroundControl via MAVLink over UDP (port 14550).
"""
import time
import math
import numpy as np
from pymavlink import mavutil

class QGCMavlinkBridge:
    def __init__(self, target_ip: str = "127.0.0.1", target_port: int = 14550, origin_lat: float = 12.9716, origin_lon: float = 77.5946, origin_alt: float = 100.0):
        """
        Initializes MAVLink UDP connection to QGroundControl.
        QGC listens on UDP port 14550 by default.
        """
        self.target_ip = target_ip
        self.target_port = target_port
        self.origin_lat = origin_lat  # Base latitude (degrees)
        self.origin_lon = origin_lon  # Base longitude (degrees)
        self.origin_alt = origin_alt  # Base altitude (meters)
        
        # Connect via MAVLink UDP output
        connection_str = f"udpout:{self.target_ip}:{self.target_port}"
        print(f"[*] Initializing MAVLink bridge to QGroundControl at {connection_str}...")
        self.mav = mavutil.mavlink_connection(connection_str, source_system=1, source_component=1)
        self.boot_time = time.time()
        self.last_heartbeat = 0.0

    def get_time_boot_ms(self) -> int:
        return int((time.time() - self.boot_time) * 1000)

    def xy_to_latlon(self, x_north: float, y_east: float):
        """Converts local NED coordinates (meters) to global WGS84 Lat/Lon."""
        # 1 degree latitude ~ 111,139 meters
        # 1 degree longitude ~ 111,139 * cos(lat) meters
        lat = self.origin_lat + (x_north / 111139.0)
        lon = self.origin_lon + (y_east / (111139.0 * math.cos(math.radians(self.origin_lat))))
        return lat, lon

    def send_heartbeat(self):
        """Sends MAVLink Heartbeat and SYS_STATUS at 1 Hz (identifying as PX4 in Autonomous Mode)."""
        now = time.time()
        if now - self.last_heartbeat >= 1.0:
            # 1. Heartbeat
            self.mav.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_FIXED_WING,
                mavutil.mavlink.MAV_AUTOPILOT_PX4,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED | mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED,
                0,
                mavutil.mavlink.MAV_STATE_ACTIVE
            )
            # 2. System Status (Required by QGC to mark vehicle healthy and display battery/sensors)
            sensors = (
                mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_GYRO |
                mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_ACCEL |
                mavutil.mavlink.MAV_SYS_STATUS_SENSOR_3D_MAG |
                mavutil.mavlink.MAV_SYS_STATUS_SENSOR_ABSOLUTE_PRESSURE |
                mavutil.mavlink.MAV_SYS_STATUS_SENSOR_GPS
            )
            self.mav.mav.sys_status_send(
                sensors, sensors, sensors,
                500,    # CPU load 50%
                12400,  # Battery 12.4V (3S LiPo)
                85,     # Battery remaining 85%
                0, 0, 0, 0, 0, 0
            )
            self.last_heartbeat = now

    def send_attitude(self, roll_rad: float, pitch_rad: float, yaw_rad: float):
        """Sends aircraft attitude (roll, pitch, yaw) to drive QGC Artificial Horizon / Compass."""
        self.mav.mav.attitude_send(
            self.get_time_boot_ms(),
            roll_rad,
            pitch_rad,
            yaw_rad,
            0.0, 0.0, 0.0  # Roll/Pitch/Yaw rates
        )

    def send_global_position(self, x_north: float, y_east: float, z_alt: float, vx: float, vy: float, vz: float, yaw_rad: float):
        """Sends fused global position (WGS84 Lat/Lon in 1e7 deg) to drive QGC Map marker."""
        lat, lon = self.xy_to_latlon(x_north, y_east)
        hdg_cdeg = int((math.degrees(yaw_rad) % 360) * 100)
        self.mav.mav.global_position_int_send(
            self.get_time_boot_ms(),
            int(lat * 1e7),
            int(lon * 1e7),
            int((self.origin_alt + z_alt) * 1000),  # AMSL altitude in mm
            int(z_alt * 1000),                      # Relative altitude in mm
            int(vx * 100),                          # Vx in cm/s
            int(vy * 100),                          # Vy in cm/s
            int(vz * 100),                          # Vz in cm/s
            hdg_cdeg
        )

    def send_gps_raw(self, x_north: float, y_east: float, z_alt: float, vel_ms: float, yaw_rad: float):
        """Sends raw GPS receiver fixes to QGC."""
        lat, lon = self.xy_to_latlon(x_north, y_east)
        hdg_cdeg = int((math.degrees(yaw_rad) % 360) * 100)
        self.mav.mav.gps_raw_int_send(
            self.get_time_boot_ms() * 1000, # microsec
            3,                               # Fix type: 3D Fix
            int(lat * 1e7),
            int(lon * 1e7),
            int((self.origin_alt + z_alt) * 1000),
            120,                             # eph (HDOP * 100)
            150,                             # epv (VDOP * 100)
            int(vel_ms * 100),               # GPS groundspeed cm/s
            hdg_cdeg,
            12                               # Satellites visible
        )

    def send_vfr_hud(self, airspeed: float, groundspeed: float, heading_deg: float, altitude: float, climb_rate: float):
        """Sends flight instrument HUD data (Airspeed, Groundspeed, Altitude, Climb rate)."""
        self.mav.mav.vfr_hud_send(
            airspeed,
            groundspeed,
            int(heading_deg % 360),
            75,                              # Throttle %
            altitude,
            climb_rate
        )

    def send_statustext(self, message: str, severity: int = mavutil.mavlink.MAV_SEVERITY_INFO):
        """Sends QGC message notifications (spoken aloud by QGC voice engine and logged in HUD)."""
        text_bytes = message.encode('utf-8')[:50]
        self.mav.mav.statustext_send(severity, text_bytes)
        print(f"[QGC STATUSTEXT] {message}")
