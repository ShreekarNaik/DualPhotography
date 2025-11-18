import numpy as np

class StateEstimator:
    """
    Kalman Filter-based State Estimator for Adaptive Compressed Sensing.
    
    Maintains the current estimate of the signal (theta_hat) and the 
    covariance matrix (Sigma) representing uncertainty.
    """
    
    def __init__(self, N, initial_sigma_scale=1.0):
        """
        Initialize the state estimator.
        
        Args:
            N (int): Dimension of the signal.
            initial_sigma_scale (float): Scaling factor for initial identity covariance.
        """
        self.N = N
        self.theta_hat = np.zeros(N)
        self.Sigma = np.eye(N) * initial_sigma_scale
        self.trace_history = [np.trace(self.Sigma)]
        
    def update(self, u_k, y_k, sigma_noise):
        """
        Update the state estimate based on a new measurement.
        
        Implements the Kalman Filter update equations:
        K_k = Sigma_k * u_k / (u_k^T * Sigma_k * u_k + sigma_noise^2)
        theta_{k+1} = theta_k + K_k * (y_k - u_k^T * theta_k)
        Sigma_{k+1} = (I - K_k * u_k^T) * Sigma_k
        
        Args:
            u_k (np.ndarray): Measurement pattern vector (N,).
            y_k (float): Scalar measurement value.
            sigma_noise (float): Standard deviation of measurement noise.
        """
        # Compute Kalman Gain
        # Numerator: Sigma * u
        Sigma_u = self.Sigma @ u_k
        
        # Denominator: u^T * Sigma * u + sigma^2 (Scalar)
        denom = np.dot(u_k, Sigma_u) + sigma_noise**2
        
        K_gain = Sigma_u / denom
        
        # Update Estimate theta_hat
        prediction_error = y_k - np.dot(u_k, self.theta_hat)
        self.theta_hat = self.theta_hat + K_gain * prediction_error
        
        # Update Covariance Sigma
        # Sigma_new = Sigma - K * u^T * Sigma
        # Note: K * u^T is an outer product (N x N) matrix
        # Efficient update: Sigma_new = Sigma - outer(K, u) @ Sigma
        # But since K = Sigma @ u / denom, we can also write:
        # Sigma_new = Sigma - (Sigma @ u @ u^T @ Sigma) / denom
        
        update_term = np.outer(K_gain, u_k) @ self.Sigma
        self.Sigma = self.Sigma - update_term
        
        # Enforce symmetry to avoid numerical issues
        self.Sigma = (self.Sigma + self.Sigma.T) / 2.0
        
        # Record trace
        self.trace_history.append(np.trace(self.Sigma))
        
    def get_estimate(self):
        """Return current signal estimate."""
        return self.theta_hat.copy()
        
    def get_covariance(self):
        """Return current covariance matrix."""
        return self.Sigma.copy()
        
    def get_uncertainty_trace(self):
        """Return the trace of the current covariance matrix."""
        return self.trace_history[-1]
