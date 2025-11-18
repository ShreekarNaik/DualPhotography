import numpy as np

def reconstruction_error(theta_hat, theta_true):
    """
    Compute L2 reconstruction error.
    
    Args:
        theta_hat (np.ndarray): Estimated signal.
        theta_true (np.ndarray): True signal.
        
    Returns:
        error (float): L2 norm of the difference.
    """
    return np.linalg.norm(theta_hat - theta_true)

def relative_error(theta_hat, theta_true):
    """
    Compute relative reconstruction error.
    
    Args:
        theta_hat (np.ndarray): Estimated signal.
        theta_true (np.ndarray): True signal.
        
    Returns:
        rel_error (float): ||theta_hat - theta_true|| / ||theta_true||
    """
    norm_true = np.linalg.norm(theta_true)
    if norm_true < 1e-10:
        return 0.0
    return np.linalg.norm(theta_hat - theta_true) / norm_true

def measurement_efficiency(error_adaptive, error_random):
    """
    Compute measurement efficiency gain.
    
    Args:
        error_adaptive (float or array): Error from adaptive CS.
        error_random (float or array): Error from random CS.
        
    Returns:
        efficiency (float or array): Ratio of random/adaptive error.
    """
    return error_random / (error_adaptive + 1e-10)
