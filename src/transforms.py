import numpy as np
import pywt
from scipy.fft import dctn, idctn

class TransformBase:
    def forward(self, image):
        raise NotImplementedError
    
    def inverse(self, coeffs, shape):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError

    def get_basis_image(self, index, shape):
        """
        Returns the basis image corresponding to the coefficient at 'index'.
        """
        # Create a one-hot vector
        # We need to know the total size. 
        # Usually shape is (H, W), so size is H*W.
        # But for Wavelets, the vector size might be different?
        # In our implementation, forward returns a flattened vector.
        # So we assume the vector size is consistent with what forward returns.
        # We might need to know the vector length.
        # Let's assume the caller knows or we can deduce it.
        # Actually, 'inverse' takes a vector.
        # We need to know the length of that vector.
        # Let's assume the vector length matches the image size for DCT/Hadamard.
        # For Wavelet, it might be slightly different due to padding.
        # We will handle this in the specific classes or pass the vector length.
        pass

class WaveletTransform(TransformBase):
    def __init__(self, wavelet='db1', levels=3):
        self.wavelet = wavelet
        self.levels = levels
        self.slices = None
        self.shape = None
        self.vec_len = None
    
    def forward(self, image):
        coeffs = pywt.wavedec2(image, self.wavelet, level=self.levels)
        vec, self.slices = pywt.coeffs_to_array(coeffs)
        self.shape = vec.shape
        self.vec_len = len(vec.flatten())
        return vec.flatten()
    
    def get_basis_image(self, index, output_shape):
        if self.vec_len is None:
             raise ValueError("Must call forward first to init dimensions.")
        vec = np.zeros(self.vec_len)
        vec[index] = 1.0
        return self.inverse(vec, output_shape)
    
    def inverse(self, vec, output_shape):
        if self.slices is None:
            raise ValueError("Must call forward first to set slices.")
        
        # Reshape if needed
        if vec.ndim == 1:
            vec = vec.reshape(self.shape)
            
        coeffs = pywt.array_to_coeffs(vec, self.slices, output_format='wavedec2')
        rec = pywt.waverec2(coeffs, self.wavelet)
        return rec[:output_shape[0], :output_shape[1]]

    def get_name(self):
        return f"Wavelet ({self.wavelet})"

class DCTTransform(TransformBase):
    def forward(self, image):
        # Type II DCT, orthonormal
        return dctn(image, norm='ortho').flatten()
    
    def inverse(self, vec, output_shape):
        arr = vec.reshape(output_shape)
        return idctn(arr, norm='ortho')

    def get_basis_image(self, index, output_shape):
        vec = np.zeros(np.prod(output_shape))
        vec[index] = 1.0
        return self.inverse(vec, output_shape)

    def get_name(self):
        return "DCT"

class HadamardTransform(TransformBase):
    """
    Implements Fast Walsh-Hadamard Transform (FWHT).
    Assumes image dimensions are powers of 2.
    """
    def _fwht_1d(self, a):
        """In-place Fast Walsh-Hadamard Transform of array a."""
        h = 1
        while h < len(a):
            for i in range(0, len(a), h * 2):
                for j in range(i, i + h):
                    x = a[j]
                    y = a[j + h]
                    a[j] = x + y
                    a[j + h] = x - y
            h *= 2
        return a

    def _ifwht_1d(self, a):
        """Inverse FWHT is just FWHT scaled by 1/N."""
        self._fwht_1d(a)
        a /= len(a)
        return a

    def forward(self, image):
        # Check if power of 2
        h, w = image.shape
        if (h & (h-1) != 0) or (w & (w-1) != 0):
            raise ValueError("Hadamard transform requires image dimensions to be powers of 2.")
        
        # Apply 1D FWHT to rows then columns
        # Copy to avoid modifying original
        data = image.astype(np.float64).copy()
        
        # Rows
        for r in range(h):
            self._fwht_1d(data[r, :])
            
        # Columns
        for c in range(w):
            self._fwht_1d(data[:, c])
            
        # Normalize? Standard Hadamard matrix is not orthonormal (H^T H = N I).
        # If we want orthonormal coefficients, we should scale by 1/N (or 1/sqrt(N) each way).
        # The _ifwht_1d above divides by N. 
        # So Forward * Inverse = (Sum) * (Sum/N) = Sum / N.
        # Wait, standard definition: H x. Inverse: 1/N H x.
        # My _fwht_1d does H x. 
        # My _ifwht_1d does 1/N H x.
        # So Inverse(Forward(x)) = 1/N H (H x) = 1/N (N I) x = x. Correct.
        
        return data.flatten()

    def inverse(self, vec, output_shape):
        h, w = output_shape
        data = vec.reshape(output_shape).copy()
        
        # Inverse is same as forward but scaled.
        # But since we defined _ifwht_1d to include scaling, we use that.
        
        # Columns
        for c in range(w):
            self._ifwht_1d(data[:, c])
            
        # Rows
        for r in range(h):
            self._ifwht_1d(data[r, :])
            
        return data

    def get_basis_image(self, index, output_shape):
        vec = np.zeros(np.prod(output_shape))
        vec[index] = 1.0
        return self.inverse(vec, output_shape)

    def get_name(self):
        return "Hadamard"
