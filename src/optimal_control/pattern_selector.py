import numpy as np
from scipy.linalg import eigh

class GreedyPatternSelector:
    """
    Selects the optimal measurement pattern using greedy information maximization.
    
    The optimal pattern is the eigenvector corresponding to the largest eigenvalue
    of the current covariance matrix Sigma.
    """
    
    def __init__(self, N):
        """
        Initialize the pattern selector.
        
        Args:
            N (int): Dimension of the signal.
        """
        self.N = N
        
    def select_pattern(self, Sigma):
        """
        Select the next measurement pattern u_k.
        
        Args:
            Sigma (np.ndarray): Current covariance matrix (N x N).
            
        Returns:
            u_k (np.ndarray): Selected pattern vector (N,).
        """
        # Compute eigendecomposition
        # eigh is optimized for Hermitian/symmetric matrices
        # It returns eigenvalues in ascending order
        eigenvalues, eigenvectors = eigh(Sigma)
        
        # Select eigenvector corresponding to the largest eigenvalue
        # This is the direction of maximum uncertainty
        u_k = eigenvectors[:, -1]
        
        # Normalize to unit energy (L2 norm = 1)
        # Eigenvectors from eigh are already normalized, but good to be safe
        norm = np.linalg.norm(u_k)
        if norm > 1e-10:
            u_k = u_k / norm
            
        return u_k
