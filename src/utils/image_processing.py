import numpy as np
from PIL import Image
import os

def load_image(path, size=None, grayscale=True):
    """
    Load an image and convert to numpy array.
    
    Args:
        path (str): Path to image file.
        size (tuple, optional): Target size (width, height).
        grayscale (bool): Convert to grayscale.
        
    Returns:
        img (np.ndarray): Image array, normalized to [0, 1].
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
        
    img = Image.open(path)
    
    if grayscale:
        img = img.convert('L')
        
    if size is not None:
        img = img.resize(size, Image.LANCZOS)
        
    img_array = np.array(img, dtype=float)
    
    # Normalize to [0, 1]
    img_array = img_array / 255.0
    
    return img_array

def compute_mse(img1, img2):
    """Mean Squared Error."""
    return np.mean((img1 - img2)**2)

def compute_psnr(img1, img2, max_val=1.0):
    """
    Peak Signal-to-Noise Ratio.
    
    Args:
        img1, img2: Images to compare.
        max_val: Maximum possible pixel value (usually 1.0 or 255).
        
    Returns:
        psnr (float): PSNR in dB.
    """
    mse = compute_mse(img1, img2)
    if mse < 1e-10:
        return 100.0  # Perfect match
    return 20 * np.log10(max_val / np.sqrt(mse))

def save_image(img_array, path):
    """Save numpy array as image."""
    # Clip to [0, 1]
    img_array = np.clip(img_array, 0, 1)
    img = Image.fromarray((img_array * 255).astype(np.uint8))
    img.save(path)
