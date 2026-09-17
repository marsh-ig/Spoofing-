"""
Layer 3 Active Trap: Normalized Cross-Correlation and Phase-Lag Detector.
"""
import numpy as np

class Layer3ActiveTrap:
    def __init__(self, min_correlation: float = 0.70, max_lag_sec: float = 0.08):
        self.min_corr = min_correlation  # Threshold for non-reactive attacker
        self.max_lag = max_lag_sec       # Threshold for smart reactive attacker (80 ms)

    def evaluate_window(self, ref_signal: np.ndarray, target_signal: np.ndarray, dt: float = 0.01, 
                        probe_window: tuple = None, max_lag_sec: float = 0.40):
        """
        Computes Normalized Cross-Correlation rho_xy(tau) over the excitation window:
        rho_xy(tau) = Cov(x_ref, y_target(t+tau)) / (sigma_x * sigma_y)
        """
        if len(ref_signal) < 20 or np.std(ref_signal) < 1e-4 or np.std(target_signal) < 1e-4:
            return 1.0, 0.0, "HEALTHY (Authentic GPS Response)", np.array([0.0]), np.array([1.0])

        if probe_window is not None:
            i_start, i_end = probe_window
        else:
            # Automatic peak excitation window detection
            ref_centered = ref_signal - np.mean(ref_signal)
            peak_loc = np.argmax(np.abs(ref_centered))
            half_win = int(0.25 / dt)
            i_start = max(0, peak_loc - half_win)
            i_end = min(len(ref_signal), peak_loc + half_win)

        x_win = ref_signal[i_start:i_end]
        x_win_zero = x_win - np.mean(x_win)
        x_norm = np.linalg.norm(x_win_zero)

        lags_sec = np.arange(-0.05, max_lag_sec + dt / 2.0, dt)
        corrs = []

        for lag in lags_sec:
            shift = int(round(lag / dt))
            y_start = i_start + shift
            y_end = i_end + shift
            
            if y_start < 0 or y_end > len(target_signal):
                corrs.append(0.0)
                continue
                
            y_win = target_signal[y_start:y_end]
            y_win_zero = y_win - np.mean(y_win)
            y_norm = np.linalg.norm(y_win_zero)
            
            if x_norm * y_norm > 1e-6:
                r = float(np.dot(x_win_zero, y_win_zero) / (x_norm * y_norm))
            else:
                r = 0.0
            corrs.append(r)

        corrs = np.array(corrs)
        peak_idx = int(np.argmax(corrs))
        peak_corr = float(corrs[peak_idx])
        estimated_lag = float(lags_sec[peak_idx])

        # Dual-Condition Decision Trigger
        if peak_corr < self.min_corr:
            status = "COMPROMISED (Non-Reactive: Missing Peak)"
        elif estimated_lag > self.max_lag:
            status = "COMPROMISED (Smart Reactive: RF Lag Detected)"
        else:
            status = "HEALTHY (Authentic GPS Response)"

        return peak_corr, estimated_lag, status, lags_sec, corrs