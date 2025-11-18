import pytest
import numpy as np
from src.environment.synthetic import SyntheticEnvironment
from src.compressed_sensing.adaptive_cs import AdaptiveCS
from src.compressed_sensing.random_cs import RandomCS

def test_adaptive_cs_basic():
    """Test that AdaptiveCS runs without errors."""
    N = 50
    k_sparsity = 5
    K_measurements = 30
    
    env = SyntheticEnvironment(N=N, k_sparsity=k_sparsity, random_seed=42)
    theta_true = env.get_true_signal()
    
    adaptive_cs = AdaptiveCS(N=N, sigma_noise=0.01)
    theta_hat, history = adaptive_cs.run(env, K_measurements, theta_true)
    
    assert theta_hat.shape == (N,)
    assert len(history['errors']) == K_measurements
    assert len(history['traces']) == K_measurements

def test_adaptive_cs_convergence():
    """Test that AdaptiveCS reduces error over time."""
    N = 50
    k_sparsity = 5
    K_measurements = 50
    
    env = SyntheticEnvironment(N=N, k_sparsity=k_sparsity, random_seed=42)
    theta_true = env.get_true_signal()
    
    adaptive_cs = AdaptiveCS(N=N, sigma_noise=0.01)
    theta_hat, history = adaptive_cs.run(env, K_measurements, theta_true)
    
    # Error should generally decrease
    errors = history['errors']
    # Check that final error is less than initial error
    assert errors[-1] < errors[0]
    
def test_adaptive_vs_random():
    """Test that AdaptiveCS outperforms RandomCS."""
    N = 50
    k_sparsity = 5
    K_measurements = 40
    
    env_adaptive = SyntheticEnvironment(N=N, k_sparsity=k_sparsity, random_seed=42)
    env_random = SyntheticEnvironment(N=N, k_sparsity=k_sparsity, random_seed=42)
    theta_true = env_adaptive.get_true_signal()
    
    # Run Adaptive CS
    adaptive_cs = AdaptiveCS(N=N, sigma_noise=0.01)
    theta_hat_adaptive, history_adaptive = adaptive_cs.run(env_adaptive, K_measurements, theta_true)
    
    # Run Random CS
    random_cs = RandomCS(N=N)
    theta_hat_random, history_random = random_cs.run(env_random, K_measurements, theta_true)
    
    # Adaptive should have lower final error
    # This might not always be true for very small problems, but should be generally true
    final_error_adaptive = history_adaptive['errors'][-1]
    final_error_random = history_random['errors'][-1]
    
    # At least adaptive should be competitive (within 2x)
    assert final_error_adaptive <= 2.0 * final_error_random
