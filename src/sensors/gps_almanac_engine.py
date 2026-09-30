"""
GPS Satellite Constellation & Keplerian Almanac Engine.
Simulates real GPS constellation orbital mechanics, ECEF coordinate transformations,
geometric Dilution of Precision (GDOP/PDOP), and physical pseudorange generation.
"""
import numpy as np
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict

# WGS84 & GPS Constants
WGS84_A = 6378137.0          # Semi-major axis of Earth (meters)
WGS84_F = 1.0 / 298.257223563 # Flattening factor
WGS84_B = WGS84_A * (1.0 - WGS84_F)
WGS84_E2 = 1.0 - (WGS84_B**2 / WGS84_A**2) # First eccentricity squared
MU_EARTH = 3.986005e14        # Earth gravitational constant (m^3/s^2)
OMEGA_EARTH = 7.2921151467e-5 # Earth rotation rate (rad/s)
SPEED_OF_LIGHT = 299792458.0  # Speed of light (m/s)

@dataclass
class AlmanacSatellite:
    prn: int                  # Satellite PRN ID (1 to 32)
    plane: int                # Orbital plane (A to F: 1 to 6)
    semi_major_axis: float    # a (meters) ~ 26,560,000 m
    eccentricity: float       # e ~ 0.001 to 0.015
    inclination: float        # i0 (radians) ~ 55 degrees = 0.9599 rad
    raan_0: float             # Omega0 (radians) Right Ascension of Ascending Node at epoch
    omega: float              # w (radians) Argument of Perigee
    mean_anomaly_0: float     # M0 (radians) Mean anomaly at epoch
    epoch_time: float         # t0e (seconds)

@dataclass
class SatelliteObservation:
    prn: int
    pos_ecef: np.ndarray      # [X, Y, Z] in meters
    vel_ecef: np.ndarray      # [Vx, Vy, Vz] in m/s
    elevation_deg: float      # Elevation angle from UAV horizon (deg)
    azimuth_deg: float        # Azimuth angle from True North (deg)
    los_unit_vector: np.ndarray # Unit vector pointing from UAV to Satellite
    true_range: float         # Geometric Euclidean distance (meters)
    pseudorange: float        # Measured pseudorange with clock bias & noise (meters)
    range_rate: float         # Geometric range rate / Doppler velocity (m/s)

class GPSAlmanacEngine:
    def __init__(self, origin_lat: float = 12.9716, origin_lon: float = 77.5946, origin_alt: float = 100.0, mask_elevation_deg: float = 10.0):
        """
        Initializes a realistic 31-satellite GPS constellation across 6 orbital planes.
        """
        self.origin_lat = origin_lat
        self.origin_lon = origin_lon
        self.origin_alt = origin_alt
        self.mask_elevation_deg = mask_elevation_deg
        
        # Base origin in ECEF
        self.origin_ecef = self.geodetic_to_ecef(origin_lat, origin_lon, origin_alt)
        
        # Build 31-satellite GPS Constellation
        self.satellites = self._generate_gps_constellation()
        
        # Receiver Clock bias (meters) & drift
        self.rx_clock_bias_m = 1500.0 # ~5 microseconds receiver clock bias
        self.rx_clock_drift_ms = 0.05 # clock drift

    def _generate_gps_constellation(self) -> List[AlmanacSatellite]:
        """Generates standard 31-satellite GPS constellation across 6 orbital planes (A-F)."""
        sats = []
        nominal_a = 26559700.0 # Nominal radius for 11h 58m semi-synchronous orbit
        nominal_i = math.radians(55.0)
        
        prn = 1
        for plane_idx in range(6):
            raan_0 = plane_idx * (2.0 * math.pi / 6.0) # Planes spaced 60 deg apart
            sats_in_plane = 5 if plane_idx < 5 else 6
            
            for sat_idx in range(sats_in_plane):
                # Satellite spacing within plane
                m0 = sat_idx * (2.0 * math.pi / sats_in_plane) + (plane_idx * math.radians(15.0))
                m0 = m0 % (2.0 * math.pi)
                e = 0.003 + (prn % 5) * 0.001
                omega = math.radians((prn * 45.0) % 360.0)
                
                sats.append(AlmanacSatellite(
                    prn=prn,
                    plane=plane_idx + 1,
                    semi_major_axis=nominal_a + (prn % 7 - 3) * 1000.0,
                    eccentricity=e,
                    inclination=nominal_i + math.radians((prn % 3 - 1) * 0.5),
                    raan_0=raan_0,
                    omega=omega,
                    mean_anomaly_0=m0,
                    epoch_time=0.0
                ))
                prn += 1
                if prn > 31:
                    break
            if prn > 31:
                break
        return sats

    @staticmethod
    def geodetic_to_ecef(lat_deg: float, lon_deg: float, alt_m: float) -> np.ndarray:
        """Converts WGS84 Geodetic Coordinates (Lat, Lon, Alt) to ECEF (X, Y, Z)."""
        lat_rad = math.radians(lat_deg)
        lon_rad = math.radians(lon_deg)
        sin_lat = math.sin(lat_rad)
        cos_lat = math.cos(lat_rad)
        sin_lon = math.sin(lon_rad)
        cos_lon = math.cos(lon_rad)
        
        N = WGS84_A / math.sqrt(1.0 - WGS84_E2 * sin_lat**2)
        
        x = (N + alt_m) * cos_lat * cos_lon
        y = (N + alt_m) * cos_lat * sin_lon
        z = (N * (1.0 - WGS84_E2) + alt_m) * sin_lat
        return np.array([x, y, z])

    def ned_to_ecef(self, uav_ned_m: np.ndarray) -> np.ndarray:
        """Converts local North-East-Down (NED) position in meters to global ECEF."""
        lat_rad = math.radians(self.origin_lat)
        lon_rad = math.radians(self.origin_lon)
        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
        
        # Rotation matrix from NED to ECEF
        R_ned_to_ecef = np.array([
            [-sin_lat * cos_lon, -sin_lon, -cos_lat * cos_lon],
            [-sin_lat * sin_lon,  cos_lon, -cos_lat * sin_lon],
            [ cos_lat,                   0.0, -sin_lat]
        ])
        
        return self.origin_ecef + (R_ned_to_ecef @ uav_ned_m)

    def compute_satellite_orbit(self, sat: AlmanacSatellite, t_sec: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solves Kepler's Equation to compute satellite ECEF position & velocity at time t.
        """
        dt = t_sec - sat.epoch_time
        n = math.sqrt(MU_EARTH / (sat.semi_major_axis**3)) # Mean motion (rad/s)
        M = (sat.mean_anomaly_0 + n * dt) % (2.0 * math.pi)
        
        # Newton-Raphson solver for Eccentric Anomaly E: E - e*sin(E) = M
        E = M
        for _ in range(10):
            f = E - sat.eccentricity * math.sin(E) - M
            f_prime = 1.0 - sat.eccentricity * math.cos(E)
            delta_E = -f / f_prime
            E += delta_E
            if abs(delta_E) < 1e-10:
                break
                
        # True anomaly nu
        sin_nu = (math.sqrt(1.0 - sat.eccentricity**2) * math.sin(E)) / (1.0 - sat.eccentricity * math.cos(E))
        cos_nu = (math.cos(E) - sat.eccentricity) / (1.0 - sat.eccentricity * math.cos(E))
        nu = math.atan2(sin_nu, cos_nu)
        
        # Radius & argument of latitude u
        r = sat.semi_major_axis * (1.0 - sat.eccentricity * math.cos(E))
        u = sat.omega + nu
        
        # Orbital plane coordinates
        x_prime = r * math.cos(u)
        y_prime = r * math.sin(u)
        
        # ECEF transformation with Earth rotation correction
        raan = sat.raan_0 - OMEGA_EARTH * dt
        cos_raan, sin_raan = math.cos(raan), math.sin(raan)
        cos_i, sin_i = math.cos(sat.inclination), math.sin(sat.inclination)
        
        sat_x = x_prime * cos_raan - y_prime * cos_i * sin_raan
        sat_y = x_prime * sin_raan + y_prime * cos_i * cos_raan
        sat_z = y_prime * sin_i
        
        # Velocity in orbit plane
        r_dot = (sat.semi_major_axis * sat.eccentricity * n * math.sin(E)) / (1.0 - sat.eccentricity * math.cos(E))
        v_u = (sat.semi_major_axis * math.sqrt(1.0 - sat.eccentricity**2) * n) / (1.0 - sat.eccentricity * math.cos(E))
        vx_prime = r_dot * math.cos(u) - v_u * math.sin(u)
        vy_prime = r_dot * math.sin(u) + v_u * math.cos(u)
        
        sat_vx = vx_prime * cos_raan - vy_prime * cos_i * sin_raan - OMEGA_EARTH * sat_y
        sat_vy = vx_prime * sin_raan + vy_prime * cos_i * cos_raan + OMEGA_EARTH * sat_x
        sat_vz = vy_prime * sin_i
        
        return np.array([sat_x, sat_y, sat_z]), np.array([sat_vx, sat_vy, sat_vz])

    def observe_constellation(self, uav_ned_m: np.ndarray, uav_vel_ned_ms: np.ndarray, t_sec: float) -> Tuple[List[SatelliteObservation], Dict[str, float]]:
        """
        Computes visible satellite geometry, line-of-sight vectors, pseudoranges, and DOP metrics.
        """
        uav_ecef = self.ned_to_ecef(uav_ned_m)
        
        # Lat/Lon for local ENU transformation
        lat_rad = math.radians(self.origin_lat)
        lon_rad = math.radians(self.origin_lon)
        sin_lat, cos_lat = math.sin(lat_rad), math.cos(lat_rad)
        sin_lon, cos_lon = math.sin(lon_rad), math.cos(lon_rad)
        
        R_ecef_to_enu = np.array([
            [-sin_lon,            cos_lon,           0.0],
            [-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat],
            [ cos_lat * cos_lon,  cos_lat * sin_lon, sin_lat]
        ])
        
        observations = []
        G_matrix_rows = []
        
        for sat in self.satellites:
            sat_pos, sat_vel = self.compute_satellite_orbit(sat, t_sec)
            
            # Line of sight vector
            diff_ecef = sat_pos - uav_ecef
            true_range = float(np.linalg.norm(diff_ecef))
            los_unit = diff_ecef / true_range
            
            # Local ENU vector to calculate Azimuth and Elevation
            enu_vec = R_ecef_to_enu @ diff_ecef
            e_dist = math.sqrt(enu_vec[0]**2 + enu_vec[1]**2)
            elevation_deg = math.degrees(math.atan2(enu_vec[2], e_dist))
            azimuth_deg = (math.degrees(math.atan2(enu_vec[0], enu_vec[1])) + 360.0) % 360.0
            
            # Check elevation mask
            if elevation_deg >= self.mask_elevation_deg:
                # Range rate (Doppler velocity projection)
                range_rate = float(np.dot(los_unit, sat_vel))
                
                # Physical delays: Ionosphere (2.5m) + Troposphere (2.3m) + Receiver noise (0.3m)
                iono_delay = 2.5 / math.sin(math.radians(max(15.0, elevation_deg)))
                tropo_delay = 2.3 / math.sin(math.radians(max(15.0, elevation_deg)))
                noise = np.random.normal(0, 0.35)
                
                pseudorange = true_range + self.rx_clock_bias_m + self.rx_clock_drift_ms * t_sec + iono_delay + tropo_delay + noise
                
                observations.append(SatelliteObservation(
                    prn=sat.prn,
                    pos_ecef=sat_pos,
                    vel_ecef=sat_vel,
                    elevation_deg=elevation_deg,
                    azimuth_deg=azimuth_deg,
                    los_unit_vector=los_unit,
                    true_range=true_range,
                    pseudorange=pseudorange,
                    range_rate=range_rate
                ))
                
                # Geometry matrix row: [-los_x, -los_y, -los_z, 1]
                # In local ENU frame:
                los_enu = enu_vec / true_range
                G_matrix_rows.append([-los_enu[0], -los_enu[1], -los_enu[2], 1.0])
                
        # Compute Dilution of Precision (DOP) metrics
        dop_metrics = {"GDOP": 2.5, "PDOP": 2.0, "HDOP": 1.2, "VDOP": 1.6, "SatsVisible": len(observations)}
        if len(G_matrix_rows) >= 4:
            G = np.array(G_matrix_rows)
            try:
                Q = np.linalg.inv(G.T @ G)
                dop_metrics["GDOP"] = float(math.sqrt(np.trace(Q)))
                dop_metrics["PDOP"] = float(math.sqrt(Q[0,0] + Q[1,1] + Q[2,2]))
                dop_metrics["HDOP"] = float(math.sqrt(Q[0,0] + Q[1,1]))
                dop_metrics["VDOP"] = float(math.sqrt(Q[2,2]))
            except np.linalg.LinAlgError:
                pass
                
        return observations, dop_metrics
