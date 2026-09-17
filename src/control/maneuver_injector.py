"""
Micro-Maneuver Probing Injector: Generates randomized control doublets.
"""
import numpy as np

class MicroManeuverInjector:
    def __init__(self, probe_times=[15.0, 35.0, 50.0], duration: float = 0.3, amplitude_deg: float = 1.0):
        self.probe_times = probe_times
        self.duration = duration
        self.amplitude_rad = np.radians(amplitude_deg)

    def get_probe_command(self, t: float):
        """
        Returns the commanded excitation pulse:
        Positive doublet half-cycle, followed by negative half-cycle.
        """
        for t_probe in self.probe_times:
            if t_probe <= t < t_probe + self.duration:
                dt = t - t_probe
                if dt < self.duration / 2.0:
                    return self.amplitude_rad, True # Excitation Active
                else:
                    return -self.amplitude_rad, True
        return 0.0, False