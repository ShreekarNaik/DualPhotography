import numpy as np

class RandomCS:
    """
    Random Compressed Sensing Baseline.
    
    Uses random Gaussian measurement patterns and standard least squares 
    (or pseudo-inverse) for reconstruction.
    
    Note: For a fair comparison with the adaptive method (which uses L2 minimization 
    via Kalman Filter), we use the same L2 reconstruction here.
    """
    
    def __init__(self, N):
        """
        Initialize Random CS.
        
        Args:
            N (int): Signal dimension.
        """
        self.N = N
        self.history = {
            'patterns': [],
            'measurements': [],
            'errors': []
        }
        
    def run(self, environment, K_measurements, theta_true=None):
        """
        Run random compressed sensing.
        
        Args:
            environment: Object with measure(u_k) method.
            K_measurements (int): Number of measurements.
            theta_true (np.ndarray, optional): Ground truth for error tracking.
            
        Returns:
            theta_hat (np.ndarray): Final signal estimate.
            history (dict): History of metrics.
        """
        # Generate all random patterns at once
        # Random Gaussian vectors normalized to unit length
        U = np.random.randn(K_measurements, self.N)
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        
        y = np.zeros(K_measurements)
        
        for k in range(K_measurements):
            u_k = U[k]
            y_k = environment.measure(u_k)
            y[k] = y_k
            
            # Record history
            self.history['patterns'].append(u_k)
            self.history['measurements'].append(y_k)
            
            if theta_true is not None:
                # Reconstruct using measurements up to k
                # theta_hat = pinv(U_k) * y_k
                U_k = U[:k+1]
                y_curr = y[:k+1]
                
                # Least squares solution
                theta_hat, _, _, _ = np.linalg.lstsq(U_k, y_curr, rcond=None)
                
                error = np.linalg.norm(theta_hat - theta_true)
                self.history['errors'].append(error)
        
        # Final reconstruction
        theta_hat, _, _, _ = np.linalg.lstsq(U, y, rcond=None)
        return theta_hat, self.history
