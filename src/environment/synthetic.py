import numpy as np

class SyntheticEnvironment:
    """
    Synthetic environment for testing adaptive compressed sensing.
    Simulates measurements of a sparse signal with noise.
    """
    
    def __init__(self, N, k_sparsity, sigma_noise=0.01, random_seed=None):
        """
        Initialize the synthetic environment.
        
        Args:
            N (int): Dimension of the signal.
            k_sparsity (int): Number of non-zero elements in the true signal.
            sigma_noise (float): Standard deviation of measurement noise.
            random_seed (int): Random seed for reproducibility.
        """
        self.N = N
        self.k_sparsity = k_sparsity
        self.sigma_noise = sigma_noise
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        self.theta_true = self._generate_sparse_signal()
        
    def _generate_sparse_signal(self):
        """Generate a k-sparse signal."""
        theta = np.zeros(self.N)
        indices = np.random.choice(self.N, self.k_sparsity, replace=False)
        # Random amplitudes from normal distribution
        theta[indices] = np.random.randn(self.k_sparsity)
        return theta
        
    def measure(self, u_k):
        """
        Take a measurement with pattern u_k.
        
        y_k = u_k^T * theta_true + noise
        
        Args:
            u_k (np.ndarray): Measurement pattern.
            
        Returns:
            y_k (float): Noisy measurement.
        """
        true_val = np.dot(u_k, self.theta_true)
        noise = np.random.randn() * self.sigma_noise
        return true_val + noise
        
    def get_true_signal(self):
        """Return the ground truth signal."""
        return self.theta_true.copy()
