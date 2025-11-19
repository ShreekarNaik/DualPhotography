import numpy as np
from utils.basis_generation import get_hadamard_matrix, get_fourier_matrix

class DeterministicCS:
    """
    Deterministic Compressed Sensing using fixed bases (Hadamard, Fourier).
    
    Instead of random patterns, this uses a fixed basis and selects patterns
    either sequentially (low frequency first) or randomly.
    """
    
    def __init__(self, N, basis_type='hadamard', sampling_strategy='random'):
        """
        Initialize Deterministic CS.
        
        Args:
            N (int): Signal dimension.
            basis_type (str): 'hadamard' or 'fourier'.
            sampling_strategy (str): 'random' (subsampling) or 'sequential' (low-freq first).
        """
        self.N = N
        self.basis_type = basis_type
        self.sampling_strategy = sampling_strategy
        
        if basis_type == 'hadamard':
            self.Basis = get_hadamard_matrix(N)
        elif basis_type == 'fourier':
            self.Basis = get_fourier_matrix(N)
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")
            
        self.history = {
            'patterns': [],
            'measurements': [],
            'errors': []
        }
        
    def run(self, environment, K_measurements, theta_true=None):
        """
        Run deterministic compressed sensing.
        
        Args:
            environment: Object with measure(u_k) method.
            K_measurements (int): Number of measurements.
            theta_true (np.ndarray, optional): Ground truth for error tracking.
            
        Returns:
            theta_hat (np.ndarray): Final signal estimate.
            history (dict): History of metrics.
        """
        # Select patterns
        if self.sampling_strategy == 'random':
            indices = np.random.choice(self.N, K_measurements, replace=False)
        elif self.sampling_strategy == 'sequential':
            # For Fourier/DCT, low indices are low frequencies
            # For Hadamard (natural order), it's mixed, but we'll assume sequential for now
            indices = np.arange(K_measurements)
        else:
            raise ValueError(f"Unknown strategy: {self.sampling_strategy}")
            
        U = self.Basis[indices]
        
        # Normalize patterns to unit energy
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
                U_k = U[:k+1]
                y_curr = y[:k+1]
                
                # Least squares solution (L2 min)
                # For CS, L1 min is better, but for fair comparison with our L2-based adaptive method
                # we use L2 here. If the basis is orthogonal and we sample K < N, 
                # L2 gives the projection onto the subspace spanned by U_k.
                theta_hat, _, _, _ = np.linalg.lstsq(U_k, y_curr, rcond=None)
                
                error = np.linalg.norm(theta_hat - theta_true)
                self.history['errors'].append(error)
        
        # Final reconstruction
        theta_hat, _, _, _ = np.linalg.lstsq(U, y, rcond=None)
        return theta_hat, self.history
