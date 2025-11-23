import numpy as np

class DiagonalKalmanFilter:
    """
    A Diagonal Kalman Filter for estimating coefficients.
    Assumes coefficients are independent (Diagonal Covariance).
    """
    def __init__(self, n_coeffs, initial_uncertainty=1.0, measurement_noise=0.1):
        self.x_hat = np.zeros(n_coeffs)  # State estimate
        self.P = np.ones(n_coeffs) * initial_uncertainty  # Diagonal Covariance
        self.R = measurement_noise  # Measurement noise variance

    def update(self, index, measurement):
        """
        Update the estimate for a single coefficient 'index' with a new 'measurement'.
        """
        # Kalman Gain
        # K = P / (P + R)
        K = self.P[index] / (self.P[index] + self.R)
        
        # State Update
        # x = x + K(y - x)
        self.x_hat[index] = self.x_hat[index] + K * (measurement - self.x_hat[index])
        
        # Covariance Update
        # P = (1 - K)P
        self.P[index] = (1 - K) * self.P[index]
        
        return self.x_hat[index]
    
    def get_estimate(self):
        return self.x_hat.copy()
