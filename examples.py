#!/usr/bin/env python3
"""
Example script demonstrating adaptive compressed sensing workflow.

This script shows how to use the adaptive CS algorithm with different
configurations and compare with random CS baseline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib.pyplot as plt

from environment.synthetic import SyntheticEnvironment
from compressed_sensing.adaptive_cs import AdaptiveCS
from compressed_sensing.random_cs import RandomCS
from utils.visualization import plot_convergence, plot_uncertainty

def example_basic():
    """Basic example with default parameters."""
    print("=" * 60)
    print("Example 1: Basic Adaptive CS")
    print("=" * 60)
    
    # Create environment
    env = SyntheticEnvironment(N=100, k_sparsity=10, sigma_noise=0.01, random_seed=42)
    theta_true = env.get_true_signal()
    
    # Run adaptive CS
    adaptive_cs = AdaptiveCS(N=100, sigma_noise=0.01)
    theta_hat, history = adaptive_cs.run(env, K_measurements=50, theta_true=theta_true)
    
    # Results
    final_error = np.linalg.norm(theta_hat - theta_true)
    print(f"Final reconstruction error: {final_error:.4f}")
    print(f"Final uncertainty (trace): {history['traces'][-1]:.4f}")
    print()

def example_comparison():
    """Compare adaptive vs random CS."""
    print("=" * 60)
    print("Example 2: Adaptive vs Random CS Comparison")
    print("=" * 60)
    
    N = 100
    k_sparsity = 10
    K_measurements = 50
    
    # Create identical environments
    env_adaptive = SyntheticEnvironment(N=N, k_sparsity=k_sparsity, random_seed=123)
    env_random = SyntheticEnvironment(N=N, k_sparsity=k_sparsity, random_seed=123)
    theta_true = env_adaptive.get_true_signal()
    
    # Run both methods
    print("Running Adaptive CS...")
    adaptive_cs = AdaptiveCS(N=N, sigma_noise=0.01)
    theta_hat_adaptive, hist_adaptive = adaptive_cs.run(env_adaptive, K_measurements, theta_true)
    
    print("Running Random CS...")
    random_cs = RandomCS(N=N)
    theta_hat_random, hist_random = random_cs.run(env_random, K_measurements, theta_true)
    
    # Compare
    error_adaptive = np.linalg.norm(theta_hat_adaptive - theta_true)
    error_random = np.linalg.norm(theta_hat_random - theta_true)
    improvement = error_random / error_adaptive
    
    print(f"\nResults:")
    print(f"  Adaptive CS error: {error_adaptive:.4f}")
    print(f"  Random CS error:   {error_random:.4f}")
    print(f"  Improvement:       {improvement:.2f}x")
    print()

def example_varying_sparsity():
    """Test performance with different sparsity levels."""
    print("=" * 60)
    print("Example 3: Performance vs. Sparsity Level")
    print("=" * 60)
    
    N = 100
    K_measurements = 40
    sparsity_levels = [5, 10, 15, 20]
    
    results = []
    
    for k in sparsity_levels:
        env = SyntheticEnvironment(N=N, k_sparsity=k, random_seed=42)
        theta_true = env.get_true_signal()
        
        adaptive_cs = AdaptiveCS(N=N, sigma_noise=0.01)
        theta_hat, history = adaptive_cs.run(env, K_measurements, theta_true)
        
        error = np.linalg.norm(theta_hat - theta_true)
        results.append((k, error))
        print(f"Sparsity k={k:2d}: Error = {error:.4f}")
    
    print()

def example_measurement_budget():
    """Show error vs number of measurements."""
    print("=" * 60)
    print("Example 4: Error vs. Measurement Budget")
    print("=" * 60)
    
    N = 100
    k_sparsity = 10
    measurement_counts = [10, 20, 30, 40, 50, 60]
    
    print("Measurements | Adaptive Error | Random Error | Improvement")
    print("-" * 60)
    
    for K in measurement_counts:
        env_adaptive = SyntheticEnvironment(N=N, k_sparsity=k_sparsity, random_seed=42)
        env_random = SyntheticEnvironment(N=N, k_sparsity=k_sparsity, random_seed=42)
        theta_true = env_adaptive.get_true_signal()
        
        # Adaptive
        adaptive_cs = AdaptiveCS(N=N, sigma_noise=0.01)
        theta_hat_adaptive, _ = adaptive_cs.run(env_adaptive, K, theta_true)
        error_adaptive = np.linalg.norm(theta_hat_adaptive - theta_true)
        
        # Random
        random_cs = RandomCS(N=N)
        theta_hat_random, _ = random_cs.run(env_random, K, theta_true)
        error_random = np.linalg.norm(theta_hat_random - theta_true)
        
        improvement = error_random / error_adaptive if error_adaptive > 0 else float('inf')
        
        print(f"    {K:3d}      |     {error_adaptive:.4f}     |    {error_random:.4f}    |    {improvement:.2f}x")
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Adaptive Compressed Sensing - Examples")
    print("=" * 60 + "\n")
    
    example_basic()
    example_comparison()
    example_varying_sparsity()
    example_measurement_budget()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)
