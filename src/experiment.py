import numpy as np
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
from estimator import DiagonalKalmanFilter

from skimage import io
import os

class Experiment:
    def __init__(self, image, transform, strategy, max_measurements=5000, log_interval=500, save_patterns_dir=None):
        self.image = image
        self.transform = transform
        self.strategy = strategy
        self.max_measurements = max_measurements
        self.log_interval = log_interval
        self.save_patterns_dir = save_patterns_dir
        
        if self.save_patterns_dir:
            os.makedirs(self.save_patterns_dir, exist_ok=True)
        
    def run(self):
        # 1. Ground Truth Transform
        true_vec = self.transform.forward(self.image)
        n_coeffs = len(true_vec)
        
        # 2. Initialize Estimator
        kf = DiagonalKalmanFilter(n_coeffs)
        
        results = {
            "measurements": [],
            "mse": [],
            "psnr": []
        }
        
        print(f"Running {self.transform.get_name()} with {self.strategy.get_name()}...")
        
        for k in range(self.max_measurements):
            # Select next measurement
            idx = self.strategy.next_index(kf.get_estimate())
            if idx is None:
                break
            
            # Save Pattern if requested
            if self.save_patterns_dir:
                # Only save first 100 or so to avoid disk explosion? 
                # User said "since they might be many", implying they want them.
                # But 3000 images is a lot. Let's save them all but maybe warn?
                # Or save every Nth?
                # Let's save ALL for now as requested.
                
                # Generate Basis Image
                basis_img = self.transform.get_basis_image(idx, self.image.shape)
                
                # Normalize for visualization (basis images can have negative values)
                # Shift and scale to 0-255
                # Basis images are often centered at 0.
                # Let's map min/max to 0-255.
                if basis_img.max() != basis_img.min():
                    norm_img = (basis_img - basis_img.min()) / (basis_img.max() - basis_img.min())
                else:
                    norm_img = np.zeros_like(basis_img)
                
                save_path = os.path.join(self.save_patterns_dir, f"pattern_{k:05d}.png")
                io.imsave(save_path, (norm_img * 255).astype(np.uint8))
            
            # Simulate Measurement (True Value + Noise)
            measurement = true_vec[idx] + np.random.normal(0, 0.1)
            
            # Update Estimator
            kf.update(idx, measurement)
            
            # Log Results
            if (k + 1) % self.log_interval == 0:
                est_vec = kf.get_estimate()
                rec_img = self.transform.inverse(est_vec, self.image.shape)
                
                # Clip to valid range
                rec_img = np.clip(rec_img, 0, 1) # Assuming normalized image
                
                mse = mean_squared_error(self.image, rec_img)
                psnr = peak_signal_noise_ratio(self.image, rec_img, data_range=1.0)
                
                results["measurements"].append(k + 1)
                results["mse"].append(mse)
                results["psnr"].append(psnr)
                # print(f"  Step {k+1}: PSNR = {psnr:.2f} dB")
                
        # Final Reconstruction
        est_vec = kf.get_estimate()
        rec_img = self.transform.inverse(est_vec, self.image.shape)
        rec_img = np.clip(rec_img, 0, 1)
        
        return results, rec_img
