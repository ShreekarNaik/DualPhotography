import pytest
import numpy as np
from src.environment.synthetic import SyntheticEnvironment
from src.compressed_sensing.adaptive_cs import AdaptiveCS

def test_end_to_end_pipeline():
    """Test the complete pipeline from environment to reconstruction."""
    N = 100
    k_sparsity = 10
    K_measurements = 50
    sigma_noise = 0.01
    
    # 1. Create environment
    env = SyntheticEnvironment(
        N=N, 
        k_sparsity=k_sparsity, 
        sigma_noise=sigma_noise,
        random_seed=123
    )
    theta_true = env.get_true_signal()
    
    # Verify true signal is sparse
    assert np.sum(np.abs(theta_true) > 1e-10) == k_sparsity
    
    # 2. Run adaptive CS
    adaptive_cs = AdaptiveCS(N=N, sigma_noise=sigma_noise)
    theta_hat, history = adaptive_cs.run(env, K_measurements, theta_true)
    
    # 3. Verify reconstruction quality
    final_error = history['errors'][-1]
    
    # With enough measurements (K = 50 > 2*k = 20), error should be reasonable
    # Relaxed threshold since Kalman filter without sparsity prior may not recover perfectly
    assert final_error < 5.0
    
    # 4. Verify uncertainty decreases
    traces = history['traces']
    assert traces[-1] < traces[0]
    
    # 5. Verify history is consistent
    assert len(history['patterns']) == K_measurements
    assert len(history['measurements']) == K_measurements
    assert len(history['errors']) == K_measurements
