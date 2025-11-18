import pytest
import numpy as np
from src.optimal_control.state_estimator import StateEstimator

def test_state_estimator_initialization():
    """Test that StateEstimator initializes correctly."""
    N = 10
    estimator = StateEstimator(N)
    
    assert estimator.N == N
    assert estimator.theta_hat.shape == (N,)
    assert estimator.Sigma.shape == (N, N)
    assert np.allclose(estimator.theta_hat, 0.0)
    assert np.allclose(estimator.Sigma, np.eye(N))

def test_state_estimator_update():
    """Test that StateEstimator update reduces uncertainty."""
    N = 10
    estimator = StateEstimator(N)
    
    # Take a measurement
    u_k = np.random.randn(N)
    u_k = u_k / np.linalg.norm(u_k)
    y_k = 1.0
    sigma_noise = 0.1
    
    initial_trace = np.trace(estimator.Sigma)
    estimator.update(u_k, y_k, sigma_noise)
    final_trace = np.trace(estimator.Sigma)
    
    # Trace should decrease (uncertainty reduces)
    assert final_trace < initial_trace
    
    # Covariance should remain symmetric
    assert np.allclose(estimator.Sigma, estimator.Sigma.T)

def test_state_estimator_convergence():
    """Test that with multiple measurements, estimate converges."""
    N = 50
    k_sparsity = 5
    
    # Create a true signal
    theta_true = np.zeros(N)
    indices = np.random.choice(N, k_sparsity, replace=False)
    theta_true[indices] = np.random.randn(k_sparsity)
    
    estimator = StateEstimator(N)
    sigma_noise = 0.01
    
    # Take many measurements
    for _ in range(100):
        u_k = np.random.randn(N)
        u_k = u_k / np.linalg.norm(u_k)
        y_k = np.dot(u_k, theta_true) + np.random.randn() * sigma_noise
        estimator.update(u_k, y_k, sigma_noise)
    
    # Check that estimate is close to true value
    error = np.linalg.norm(estimator.theta_hat - theta_true)
    # With 100 measurements, error should be small
    assert error < 1.0  # Generous threshold
