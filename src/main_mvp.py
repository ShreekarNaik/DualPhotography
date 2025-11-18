import argparse
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.synthetic import SyntheticEnvironment
from compressed_sensing.adaptive_cs import AdaptiveCS
from compressed_sensing.random_cs import RandomCS
from utils.visualization import plot_convergence, plot_uncertainty, visualize_patterns

def main():
    parser = argparse.ArgumentParser(description='Run MVP Synthetic Tests for Adaptive CS')
    parser.add_argument('--N', type=int, default=100, help='Signal dimension')
    parser.add_argument('--sparsity', type=int, default=5, help='Sparsity level')
    parser.add_argument('--measurements', type=int, default=50, help='Number of measurements')
    parser.add_argument('--noise', type=float, default=0.01, help='Noise standard deviation')
    parser.add_argument('--output', type=str, default='results', help='Output directory for plots')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    print(f"Running MVP Test with N={args.N}, Sparsity={args.sparsity}, K={args.measurements}")
    
    # 1. Setup Environment
    env = SyntheticEnvironment(
        N=args.N, 
        k_sparsity=args.sparsity, 
        sigma_noise=args.noise,
        random_seed=args.seed
    )
    theta_true = env.get_true_signal()
    
    # 2. Run Adaptive CS
    print("Running Adaptive Compressed Sensing...")
    adaptive_cs = AdaptiveCS(N=args.N, sigma_noise=args.noise)
    theta_hat_adaptive, history_adaptive = adaptive_cs.run(
        env, 
        K_measurements=args.measurements, 
        theta_true=theta_true
    )
    
    final_error_adaptive = np.linalg.norm(theta_hat_adaptive - theta_true)
    print(f"Adaptive CS Final Error: {final_error_adaptive:.6f}")
    
    # 3. Run Random CS (Baseline)
    print("Running Random Compressed Sensing (Baseline)...")
    random_cs = RandomCS(N=args.N)
    theta_hat_random, history_random = random_cs.run(
        env, 
        K_measurements=args.measurements, 
        theta_true=theta_true
    )
    
    final_error_random = np.linalg.norm(theta_hat_random - theta_true)
    print(f"Random CS Final Error: {final_error_random:.6f}")
    
    # 4. Visualize Results
    os.makedirs(args.output, exist_ok=True)
    
    plot_convergence(
        history_adaptive, 
        history_random, 
        save_path=os.path.join(args.output, 'convergence.png')
    )
    
    plot_uncertainty(
        history_adaptive, 
        save_path=os.path.join(args.output, 'uncertainty.png')
    )
    
    # Visualize patterns (only if N is square, e.g., 64, 100)
    if int(np.sqrt(args.N))**2 == args.N:
        visualize_patterns(
            history_adaptive['patterns'], 
            args.N, 
            save_path=os.path.join(args.output, 'patterns.png')
        )
        
    print(f"\nResults saved to {args.output}/")
    
    # Summary
    improvement = final_error_random / final_error_adaptive if final_error_adaptive > 0 else float('inf')
    print(f"\nSummary:")
    print(f"Adaptive CS Error: {final_error_adaptive:.6f}")
    print(f"Random CS Error:   {final_error_random:.6f}")
    print(f"Improvement Factor: {improvement:.2f}x")

if __name__ == "__main__":
    main()
