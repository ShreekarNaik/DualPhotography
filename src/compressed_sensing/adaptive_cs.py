import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimal_control.state_estimator import StateEstimator
from optimal_control.pattern_selector import GreedyPatternSelector

class AdaptiveCS:
    """
    Adaptive Compressed Sensing Algorithm.
    
    Orchestrates the loop:
    1. Select optimal pattern u_k based on current uncertainty Sigma_k
    2. Take measurement y_k
    3. Update state estimate (theta_hat, Sigma_k)
    """
    
    def __init__(self, N, sigma_noise):
        """
        Initialize the Adaptive CS algorithm.
        
        Args:
            N (int): Signal dimension.
            sigma_noise (float): Assumed noise standard deviation for Kalman filter.
        """
        self.N = N
        self.sigma_noise = sigma_noise
        
        self.estimator = StateEstimator(N)
        self.selector = GreedyPatternSelector(N)
        
        self.history = {
            'patterns': [],
            'measurements': [],
            'errors': [],
            'traces': []
        }
        
    def run(self, environment, K_measurements, theta_true=None):
        """
        Run the adaptive sensing loop.
        
        Args:
            environment: Object with measure(u_k) method.
            K_measurements (int): Number of measurements to take.
            theta_true (np.ndarray, optional): Ground truth for error tracking.
            
        Returns:
            theta_hat (np.ndarray): Final signal estimate.
            history (dict): History of metrics.
        """
        for k in range(K_measurements):
            # 1. Optimal Control: Select pattern
            Sigma = self.estimator.get_covariance()
            u_k = self.selector.select_pattern(Sigma)
            
            # 2. Measurement: Interact with environment
            y_k = environment.measure(u_k)
            
            # 3. State Update: Kalman Filter
            self.estimator.update(u_k, y_k, self.sigma_noise)
            
            # Record history
            self.history['patterns'].append(u_k)
            self.history['measurements'].append(y_k)
            self.history['traces'].append(self.estimator.get_uncertainty_trace())
            
            if theta_true is not None:
                theta_hat = self.estimator.get_estimate()
                error = np.linalg.norm(theta_hat - theta_true)
                self.history['errors'].append(error)
                
        return self.estimator.get_estimate(), self.history
