import numpy as np
import pywt
from scipy.fft import dct, idct

class TransformAdaptiveCS:
    """
    Adaptive Compressed Sensing in Transform Domain (Wavelet/DCT).
    
    Uses a Diagonal Kalman Filter to estimate transform coefficients.
    Suitable for large images (e.g., 1024x1024) as it avoids O(N^2) covariance.
    """
    
    def __init__(self, N, transform_type='wavelet', wavelet_name='db4', sigma_noise=0.01):
        """
        Initialize Transform Adaptive CS.
        
        Args:
            N (int): Total number of pixels (must be square, e.g., 1024*1024).
            transform_type (str): 'wavelet' or 'dct'.
            wavelet_name (str): Name of wavelet to use (e.g., 'db4', 'haar').
            sigma_noise (float): Measurement noise standard deviation.
        """
        self.N = N
        self.side_length = int(np.sqrt(N))
        if self.side_length * self.side_length != N:
            raise ValueError(f"N={N} must be a perfect square for image processing.")
            
        self.transform_type = transform_type
        self.wavelet_name = wavelet_name
        self.sigma_noise = sigma_noise
        
        # Precompute wavelet bookkeeping if needed
        if self.transform_type == 'wavelet':
            # Get coefficient slices/shapes
            coeffs = pywt.wavedec2(np.zeros((self.side_length, self.side_length)), self.wavelet_name)
            self.coeffs_flat_template, self.coeff_slices = pywt.coeffs_to_array(coeffs)
            self.coeffs_shape = self.coeffs_flat_template.shape
            self.num_coeffs = self.coeffs_flat_template.size
        else:
            self.num_coeffs = N
            
        # State: Estimate of transform coefficients
        self.theta_hat = np.zeros(self.num_coeffs)
        
        # Covariance: Diagonal variances only (vector of size N)
        # Initialize with a heuristic: Low frequencies have higher variance (uncertainty/energy)
        self.variances = self._initialize_variances()
        
        # History
        self.history = {
            'measurements': [],
            'patterns': [], # We store indices, not full patterns to save memory
            'variances_trace': []
        }
            
    def _initialize_variances(self):
        """
        Initialize variances with a power-law decay to prioritize low frequencies.
        """
        # Simple heuristic: Decay from low index to high index
        indices = np.arange(1, self.num_coeffs + 1)
        # Power law decay: 1 / k^alpha
        variances = 1.0 / (indices ** 1.5)
        # Normalize max variance
        variances = variances / variances.max()
        return variances
        
    def _get_basis_image(self, index):
        """
        Generate the basis image (pixel domain) for a single coefficient index.
        """
        theta_basis = np.zeros(self.num_coeffs)
        theta_basis[index] = 1.0
        
        if self.transform_type == 'dct':
            # 2D DCT Basis
            coeff_img = theta_basis.reshape((self.side_length, self.side_length))
            basis_img = idct(idct(coeff_img.T, norm='ortho').T, norm='ortho')
            return basis_img.flatten()
            
        elif self.transform_type == 'wavelet':
            # Inverse Wavelet Transform
            # Reshape 1D vector to 2D stitched array
            theta_reshaped = theta_basis.reshape(self.coeffs_shape)
            coeffs = pywt.array_to_coeffs(theta_reshaped, self.coeff_slices, output_format='wavedec2')
            basis_img = pywt.waverec2(coeffs, self.wavelet_name)
            
            # Handle size mismatch due to padding
            if basis_img.shape[0] != self.side_length:
                basis_img = basis_img[:self.side_length, :self.side_length]
                
            return basis_img.flatten()
            
    def _inverse_transform(self, theta):
        """Convert coefficients to image."""
        if self.transform_type == 'dct':
            coeff_img = theta.reshape((self.side_length, self.side_length))
            img = idct(idct(coeff_img.T, norm='ortho').T, norm='ortho')
            return img.flatten()
        elif self.transform_type == 'wavelet':
            theta_reshaped = theta.reshape(self.coeffs_shape)
            coeffs = pywt.array_to_coeffs(theta_reshaped, self.coeff_slices, output_format='wavedec2')
            img = pywt.waverec2(coeffs, self.wavelet_name)
            if img.shape[0] != self.side_length:
                img = img[:self.side_length, :self.side_length]
            return img.flatten()
            
    def run(self, environment, K_measurements, theta_true_pixels=None):
        """
        Run the adaptive loop.
        
        Args:
            environment: Object with measure(u) method.
            K_measurements: Number of measurements.
            theta_true_pixels: Ground truth image (flattened) for error tracking.
        """
        # Initialize history
        self.history['errors'] = []
        
        # Pre-calculate true coefficients if available (for debugging/analysis)
        theta_true_coeffs = None
        if theta_true_pixels is not None:
            # Forward transform
            img_2d = theta_true_pixels.reshape((self.side_length, self.side_length))
            if self.transform_type == 'dct':
                coeffs = dct(dct(img_2d.T, norm='ortho').T, norm='ortho')
                theta_true_coeffs = coeffs.flatten()
            elif self.transform_type == 'wavelet':
                coeffs = pywt.wavedec2(img_2d, self.wavelet_name)
                theta_true_coeffs_2d, _ = pywt.coeffs_to_array(coeffs)
                theta_true_coeffs = theta_true_coeffs_2d.flatten()
                
                # Ensure size match (if theta_true_coeffs is smaller/larger than expected)
                if len(theta_true_coeffs) != self.num_coeffs:
                    # This shouldn't happen if initialized correctly, but safety first
                    # Pad or crop
                    if len(theta_true_coeffs) < self.num_coeffs:
                        theta_true_coeffs = np.pad(theta_true_coeffs, (0, self.num_coeffs - len(theta_true_coeffs)))
                    else:
                        theta_true_coeffs = theta_true_coeffs[:self.num_coeffs]
        
        print(f"Starting Transform Adaptive CS ({self.transform_type})...")
        
        from tqdm import tqdm
        for k in tqdm(range(K_measurements), desc="Adaptive Measurements"):
            # 1. Select Pattern: Index with max variance
            # We mask indices we've already measured to avoid measuring same coeff twice
            # (In standard KF, variance drops, so we wouldn't pick it again anyway, 
            # but explicit masking is safer for single-coeff measurements)
            
            # Find max variance
            best_idx = np.argmax(self.variances)
            
            # 2. Generate Pattern (Basis Function)
            u_k_pixels = self._get_basis_image(best_idx)
            
            # Normalize pattern energy? 
            # Basis functions usually have unit energy in transform domain.
            # In pixel domain, if transform is orthogonal, energy is preserved.
            # Let's ensure unit norm for stability
            norm = np.linalg.norm(u_k_pixels)
            if norm > 1e-10:
                u_k_pixels = u_k_pixels / norm
                
            # 3. Measure
            y_k = environment.measure(u_k_pixels)
            
            # 4. Update State (Scalar Kalman Filter)
            # We measured a single coefficient (approximately).
            # If u_k corresponds exactly to basis vector e_i, then we measured theta_i directly.
            # y_k = theta_i + noise
            # Update equations for scalar x and measurement z = x + v:
            # K = var / (var + noise_var)
            # x_new = x + K * (z - x)
            # var_new = (1 - K) * var
            
            # Note: If u_k was normalized, we need to account for that scaling.
            # We assumed we measured the coefficient. 
            # The measurement is y = <u_pixels, image_pixels> + noise
            # <u_pixels, image_pixels> = <Inverse(e_i), Inverse(theta)>
            # If transform is orthogonal (Parseval's), this equals <e_i, theta> = theta_i.
            # So y_k is indeed a noisy measurement of theta[best_idx].
            
            # However, wavelets/DCT might not be perfectly orthonormal in implementation 
            # or if we normalized u_k.
            # Let's assume y_k = scale * theta[best_idx] + noise
            # scale = norm we divided by.
            
            # Wait, if we normalized u_k, then we measured (1/norm) * theta_i.
            # So y_k = (1/norm) * theta_i + noise
            # Let H = 1/norm.
            # K = P * H / (H*P*H + R)
            # x = x + K * (y - H*x)
            # P = (1 - K*H) * P
            
            H = 1.0 / norm if norm > 1e-10 else 1.0
            R = self.sigma_noise ** 2
            P = self.variances[best_idx]
            
            # Kalman Gain
            S = H * P * H + R  # Innovation covariance
            K = P * H / S
            
            # Innovation
            prediction = H * self.theta_hat[best_idx]
            innovation = y_k - prediction
            
            # Update
            self.theta_hat[best_idx] = self.theta_hat[best_idx] + K * innovation
            self.variances[best_idx] = (1.0 - K * H) * P
            
            # Record
            self.history['measurements'].append(y_k)
            self.history['patterns'].append(best_idx)
            self.history['variances_trace'].append(np.sum(self.variances))
            
            # Error tracking (optional, expensive to reconstruct every step)
            if theta_true_pixels is not None and k % 100 == 0:
                # Calculate error in pixel domain
                # Reconstruct current estimate
                # img_hat = self._inverse_transform(self.theta_hat)
                # error = np.linalg.norm(img_hat - theta_true_pixels)
                
                # Or error in transform domain (Parseval's -> same error)
                error = np.linalg.norm(self.theta_hat - theta_true_coeffs)
                self.history['errors'].append(error)
                
        # Final reconstruction
        img_hat = self._inverse_transform(self.theta_hat)
        return img_hat, self.history
