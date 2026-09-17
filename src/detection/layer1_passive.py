"""
Layer 1 Passive Detector: Chi-Square Mahalanobis Distance Test on EKF Innovation Residuals.
"""
import numpy as np

class Layer1PassiveDetector:
    def __init__(self, chi2_threshold: float = 22.46):
        # Chi-Square statistical threshold for alpha = 0.001 (99.9% confidence) with 6 DOF
        self.threshold = chi2_threshold

    def evaluate(self, gamma: np.ndarray, S: np.ndarray):
        """
        Computes the normalized Mahalanobis distance:
        d_M^2 = gamma^T * S^-1 * gamma
        """
        try:
            inv_S = np.linalg.inv(S)
            d_M_sq = float(gamma.T @ inv_S @ gamma)
        except np.linalg.LinAlgError:
            d_M_sq = 0.0

        is_alarm = d_M_sq > self.threshold
        status = "SUSPICIOUS" if is_alarm else "HEALTHY"
        
        return d_M_sq, is_alarm, status