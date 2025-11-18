"""
Advanced Blender integration script for Adaptive Dual Photography.

This script demonstrates how to use the Blender environment with the
adaptive compressed sensing algorithm. It must be run from within Blender.

Usage:
    blender --background --python blender_example.py -- --measurements 50 --resolution 64

Note: Requires Blender 3.0+ with Python API (bpy)
"""

import sys
import os

# Check if running in Blender
try:
    import bpy
    BLENDER_AVAILABLE = True
except ImportError:
    print("ERROR: This script must be run from within Blender!")
    print("Usage: blender --background --python blender_example.py")
    sys.exit(1)

# Add project src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

import numpy as np
from environment.blender_env import BlenderEnvironment
from compressed_sensing.adaptive_cs import AdaptiveCS

def run_blender_adaptive_cs(resolution=64, K_measurements=30, scene_name="simple"):
    """
    Run adaptive compressed sensing with Blender environment.
    
    Args:
        resolution (int): Resolution for patterns and camera
        K_measurements (int): Number of measurements to take
        scene_name (str): Scene type ('simple', 'cube')
    """
    print(f"\n{'='*60}")
    print(f"Blender Adaptive CS Demo")
    print(f"{'='*60}\n")
    print(f"Resolution: {resolution}x{resolution}")
    print(f"Measurements: {K_measurements}")
    print(f"Scene: {scene_name}\n")
    
    # 1. Setup Blender environment
    print("Setting up Blender scene...")
    env = BlenderEnvironment(resolution=resolution, scene_name=scene_name)
    env.setup_scene()
    
    # 2. Initialize adaptive CS
    N = resolution * resolution
    sigma_noise = 0.01  # Assumed noise level
    
    print(f"Initializing Adaptive CS (N={N})...")
    adaptive_cs = AdaptiveCS(N=N, sigma_noise=sigma_noise)
    
    # 3. Run adaptive sensing loop
    print(f"\nRunning adaptive sensing loop...")
    print("Measurement | Uncertainty (trace)")
    print("-" * 40)
    
    for k in range(K_measurements):
        # Select pattern
        Sigma = adaptive_cs.estimator.get_covariance()
        u_k = adaptive_cs.selector.select_pattern(Sigma)
        
        # Project and measure (through Blender)
        env.project_pattern(u_k)
        y_k = env.capture_measurement()
        
        # Update state
        adaptive_cs.estimator.update(u_k, y_k, sigma_noise)
        
        # Track progress
        trace = adaptive_cs.estimator.get_uncertainty_trace()
        
        if k % 5 == 0:  # Print every 5 measurements
            print(f"    {k:3d}     | {trace:.4e}")
    
    # 4. Get final estimate
    theta_hat = adaptive_cs.estimator.get_estimate()
    final_trace = adaptive_cs.estimator.get_uncertainty_trace()
    
    print(f"\n{'='*60}")
    print(f"Final Results:")
    print(f"  Final uncertainty: {final_trace:.4e}")
    print(f"  Estimate norm: {np.linalg.norm(theta_hat):.4f}")
    print(f"{'='*60}\n")
    
    # 5. Cleanup
    env.cleanup()
    
    print("Done!")
    

if __name__ == "__main__":
    # Parse command line arguments (after --)
    import argparse
    
    # Blender passes args after '--'
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1:]
    else:
        argv = []
    
    parser = argparse.ArgumentParser(description='Blender Adaptive CS Example')
    parser.add_argument('--resolution', type=int, default=64, help='Pattern resolution')
    parser.add_argument('--measurements', type=int, default=30, help='Number of measurements')
    parser.add_argument('--scene', type=str, default='simple', choices=['simple', 'cube'], 
                       help='Scene type')
    
    args = parser.parse_args(argv)
    
    # Run
    run_blender_adaptive_cs(
        resolution=args.resolution,
        K_measurements=args.measurements,
        scene_name=args.scene
    )
