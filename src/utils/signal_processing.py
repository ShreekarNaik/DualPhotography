import numpy as np
from scipy.fft import dct, idct

def generate_sparse_signal(N, k_sparsity, amplitude_range=(-1, 1), random_seed=None):
    """
    Generate a k-sparse signal.
    
    Args:
        N (int): Signal dimension.
        k_sparsity (int): Number of non-zero elements.
        amplitude_range (tuple): Range for non-zero amplitudes.
        random_seed (int, optional): Random seed.
        
    Returns:
        theta (np.ndarray): Sparse signal of dimension N.
    """
    if random_seed is not None:
        np.random.seed(random_seed)
        
    theta = np.zeros(N)
    indices = np.random.choice(N, k_sparsity, replace=False)
    amplitudes = np.random.uniform(amplitude_range[0], amplitude_range[1], k_sparsity)
    theta[indices] = amplitudes
    return theta

def get_sparsifying_basis(N, basis_type='identity'):
    """
    Get a sparsifying basis matrix.
    
    Args:
        N (int): Signal dimension.
        basis_type (str): Type of basis ('identity', 'dct', 'random').
        
    Returns:
        Psi (np.ndarray): N x N basis matrix.
    """
    if basis_type == 'identity':
        return np.eye(N)
    elif basis_type == 'dct':
        # DCT basis - signals sparse in frequency domain
        # Create DCT matrix by applying DCT to each standard basis vector
        Psi = np.zeros((N, N))
        for i in range(N):
            e_i = np.zeros(N)
            e_i[i] = 1.0
            Psi[:, i] = dct(e_i, norm='ortho')
        return Psi
    elif basis_type == 'random':
        # Random orthonormal basis
        A = np.random.randn(N, N)
        Q, _ = np.linalg.qr(A)
        return Q
    else:
        raise ValueError(f"Unknown basis type: {basis_type}")

def add_noise(signal, snr_db):
    """
    Add Gaussian noise to achieve a target SNR.
    
    Args:
        signal (np.ndarray): Clean signal.
        snr_db (float): Target signal-to-noise ratio in dB.
        
    Returns:
        noisy_signal (np.ndarray): Signal with added noise.
    """
    signal_power = np.mean(signal**2)
    snr_linear = 10**(snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = np.random.randn(len(signal)) * np.sqrt(noise_power)
    return signal + noise
