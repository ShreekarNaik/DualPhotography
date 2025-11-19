import pywt
import numpy as np

def debug_wavelet():
    print(f"PyWavelets version: {pywt.__version__}")
    
    # Create dummy image
    img = np.zeros((32, 32))
    
    # Decompose
    coeffs = pywt.wavedec2(img, 'db4')
    
    # Flatten
    coeffs_flat, slices = pywt.coeffs_to_array(coeffs)
    print(f"Flat array shape: {coeffs_flat.shape}")
    print(f"Slices type: {type(slices)}")
    print(f"Number of slice elements: {len(slices)}")
    print(f"First slice: {slices[0]}")
    
    # Try to reconstruct
    try:
        coeffs_rec = pywt.array_to_coeffs(coeffs_flat, slices, output_format='wavedec2')
        print("Reconstruction successful!")
    except Exception as e:
        print(f"Reconstruction failed: {e}")
        
    # Check if slice works on 1D array
    try:
        print(f"Testing slice on 1D array: {coeffs_flat[slices[0]]}")
    except Exception as e:
        print(f"Slice failed on 1D array: {e}")

if __name__ == "__main__":
    debug_wavelet()
