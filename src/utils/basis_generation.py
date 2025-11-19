import numpy as np
from scipy.linalg import hadamard

def get_hadamard_matrix(N):
    """
    Generate Hadamard matrix of size N x N.
    N must be a power of 2.
    
    Args:
        N (int): Size of matrix (must be power of 2).
        
    Returns:
        H (np.ndarray): N x N Hadamard matrix.
    """
    # Check if N is power of 2
    if (N & (N-1) != 0) or N == 0:
        # Find next power of 2
        next_pow2 = 1 << (N-1).bit_length()
        print(f"Warning: N={N} is not a power of 2. Using next power of 2: {next_pow2}")
        H = hadamard(next_pow2)
        # Crop to N x N (not ideal but works for non-power-of-2 sizes)
        return H[:N, :N]
    
    return hadamard(N)

def get_fourier_matrix(N):
    """
    Generate Fourier basis matrix of size N x N.
    
    Args:
        N (int): Size of matrix.
        
    Returns:
        F (np.ndarray): N x N Fourier matrix (real representation).
    """
    # Create DFT matrix
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    
    # Convert to real representation for projection
    # We can use real and imaginary parts as separate patterns
    # Or use DCT which is real by default
    # Here we use DCT-II which is standard for image compression
    from scipy.fft import dct
    F = np.zeros((N, N))
    for i in range(N):
        e_i = np.zeros(N)
        e_i[i] = 1.0
        F[i, :] = dct(e_i, norm='ortho')
        
    return F

def get_random_matrix(N):
    """
    Generate random Gaussian matrix.
    
    Args:
        N (int): Size of matrix.
        
    Returns:
        R (np.ndarray): N x N random matrix.
    """
    R = np.random.randn(N, N)
    # Normalize rows
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    return R
