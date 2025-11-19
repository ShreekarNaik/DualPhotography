#!/usr/bin/env python3
"""
Real Image Validation Script.

Compares Adaptive CS against Random, Hadamard, and Fourier baselines
using real images from the data/ directory.
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from environment.synthetic import SyntheticEnvironment
from compressed_sensing.adaptive_cs import AdaptiveCS
from compressed_sensing.transform_cs import TransformAdaptiveCS
from compressed_sensing.random_cs import RandomCS
from compressed_sensing.deterministic_cs import DeterministicCS
from utils.image_processing import load_image, save_image, compute_psnr, compute_mse
from utils.visualization import plot_convergence

def run_validation(image_path, resolution=32, max_measurements=0.5, output_dir='results/real_images', method='standard', transform_type='wavelet', methods_to_run=None):
    """
    Run validation on a single image.
    
    Args:
        image_path (str): Path to image.
        resolution (int): Resize image to resolution x resolution.
        max_measurements (float): Max measurements as fraction of total pixels (0.0 to 1.0).
        output_dir (str): Directory to save results.
    """
    if methods_to_run is None:
        methods_to_run = ['adaptive', 'random', 'hadamard', 'fourier']

    filename = os.path.basename(image_path)
    name_no_ext = os.path.splitext(filename)[0]
    
    print(f"\nProcessing {filename} at {resolution}x{resolution}...")
    
    # 1. Load and preprocess image
    img = load_image(image_path, size=(resolution, resolution))
    N = resolution * resolution
    theta_true = img.flatten()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    save_image(img, os.path.join(output_dir, f"{name_no_ext}_original.png"))
    
    # 2. Setup Experiment
    K_total = int(N * max_measurements)
    sigma_noise = 0.01
    
    # Create a dummy environment that returns measurements from the true image
    # The user's snippet replaces this with SyntheticEnvironment, but SyntheticEnvironment
    # expects a true signal and noise_std, which is consistent with the original setup.
    # Keeping the original ImageEnvironment for consistency with the provided context.
    class ImageEnvironment:
        def __init__(self, theta, noise_std):
            self.theta = theta
            self.noise_std = noise_std
            
        def measure(self, u):
            return np.dot(u, self.theta) + np.random.randn() * self.noise_std
            
    env = ImageEnvironment(theta_true, sigma_noise)
    
    results = {}
    
    # 3. Run Adaptive CS
    if 'adaptive' in methods_to_run:
        print(f"  Running Adaptive CS (K={K_total})...")
        
        if method == 'standard':
            adaptive_cs = AdaptiveCS(N=N, sigma_noise=sigma_noise)
            theta_hat_adaptive, hist_adaptive = adaptive_cs.run(env, K_total, theta_true)
        elif method == 'transform':
            adaptive_cs = TransformAdaptiveCS(N=N, transform_type=transform_type, sigma_noise=sigma_noise)
            theta_hat_adaptive, hist_adaptive = adaptive_cs.run(env, K_total, theta_true)
            
        results['Adaptive'] = {
            'final_image': theta_hat_adaptive.reshape(resolution, resolution),
            'errors': hist_adaptive['errors']
        }
    
    # 4. Run Random CS
    if 'random' in methods_to_run:
        print(f"  Running Random CS...")
        random_cs = RandomCS(N=N)
        theta_hat_random, hist_random = random_cs.run(env, K_total, theta_true)
        results['Random'] = {
            'final_image': theta_hat_random.reshape(resolution, resolution),
            'errors': hist_random['errors']
        }
    
    # 5. Run Hadamard CS
    if 'hadamard' in methods_to_run:
        print(f"  Running Hadamard CS...")
        hadamard_cs = DeterministicCS(N=N, basis_type='hadamard', sampling_strategy='random')
        theta_hat_hadamard, hist_hadamard = hadamard_cs.run(env, K_total, theta_true)
        results['Hadamard'] = {
            'final_image': theta_hat_hadamard.reshape(resolution, resolution),
            'errors': hist_hadamard['errors']
        }
    
    # 6. Run Fourier CS
    if 'fourier' in methods_to_run:
        print(f"  Running Fourier CS...")
        fourier_cs = DeterministicCS(N=N, basis_type='fourier', sampling_strategy='random')
        theta_hat_fourier, hist_fourier = fourier_cs.run(env, K_total, theta_true)
        results['Fourier'] = {
            'final_image': theta_hat_fourier.reshape(resolution, resolution),
            'errors': hist_fourier['errors']
        }
    
    # 7. Process Results (Convert L2 error to PSNR)
    plt.figure(figsize=(10, 6))
    
    for method_name, data in results.items():
        # Save final image
        save_image(data['final_image'], os.path.join(output_dir, f"{name_no_ext}_{method_name}.png"))
        
        # Calculate PSNR history
        errors = np.array(data['errors'])
        if len(errors) > 0:
            # MSE = error^2 / N
            mses = (errors ** 2) / N
            # Avoid log(0)
            mses = np.maximum(mses, 1e-10)
            psnrs = 20 * np.log10(1.0 / np.sqrt(mses))
            
            plt.plot(psnrs, label=method_name)
            print(f"  {method_name} Final PSNR: {psnrs[-1]:.2f} dB")
        else:
            print(f"  {method_name} Final PSNR: N/A (no history)")
        
    plt.xlabel('Measurements')
    plt.ylabel('PSNR (dB)')
    plt.title(f'Reconstruction Quality: {filename}')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, f"{name_no_ext}_psnr.png"))
    plt.close()
    
    print(f"Done. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real Image Validation')
    parser.add_argument('--image', type=str, help='Path to specific image')
    parser.add_argument('--dir', type=str, default='data', help='Directory of images')
    parser.add_argument('--res', type=int, default=32, help='Resolution to resize to')
    parser.add_argument('--ratio', type=float, default=0.6, help='Measurement ratio (K/N)')
    parser.add_argument('--method', type=str, default='standard', choices=['standard', 'transform'], help='Adaptive CS method')
    parser.add_argument('--transform', type=str, default='wavelet', choices=['wavelet', 'dct'], help='Transform type for transform method')
    parser.add_argument('--methods', type=str, default='all', help='Comma-separated list of methods to run (adaptive,random,hadamard,fourier) or "all"')
    
    args = parser.parse_args()
    
    methods_to_run = []
    if args.methods == 'all':
        methods_to_run = ['adaptive', 'random', 'hadamard', 'fourier']
    else:
        methods_to_run = args.methods.split(',')
    
    if args.image:
        run_validation(args.image, args.res, args.ratio, method=args.method, transform_type=args.transform, methods_to_run=methods_to_run)
    else:
        # Process all images in directory
        if not os.path.exists(args.dir):
            print(f"Directory {args.dir} not found.")
            sys.exit(1)
            
        files = [f for f in os.listdir(args.dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not files:
            print(f"No images found in {args.dir}")
            sys.exit(1)
            
        for f in files:
            run_validation(os.path.join(args.dir, f), args.res, args.ratio, method=args.method, transform_type=args.transform, methods_to_run=methods_to_run)
