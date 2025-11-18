import matplotlib.pyplot as plt
import numpy as np
import os

def plot_convergence(adaptive_history, random_history=None, save_path=None):
    """
    Plot reconstruction error convergence.
    
    Args:
        adaptive_history (dict): History from AdaptiveCS.
        random_history (dict, optional): History from RandomCS.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    # Plot Adaptive CS error
    errors_adaptive = adaptive_history['errors']
    plt.semilogy(errors_adaptive, label='Adaptive CS', linewidth=2)
    
    # Plot Random CS error if provided
    if random_history:
        errors_random = random_history['errors']
        # Ensure lengths match for plotting if needed, or just plot what we have
        plt.semilogy(errors_random, label='Random CS', linestyle='--', linewidth=2)
        
    plt.xlabel('Number of Measurements')
    plt.ylabel('Reconstruction Error (L2 Norm)')
    plt.title('Convergence Comparison: Adaptive vs Random CS')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def plot_uncertainty(adaptive_history, save_path=None):
    """
    Plot uncertainty reduction (trace of covariance).
    
    Args:
        adaptive_history (dict): History from AdaptiveCS.
        save_path (str, optional): Path to save the plot.
    """
    plt.figure(figsize=(10, 6))
    
    traces = adaptive_history['traces']
    plt.semilogy(traces, color='purple', linewidth=2)
    
    plt.xlabel('Number of Measurements')
    plt.ylabel('Trace of Covariance Matrix')
    plt.title('Uncertainty Reduction (Adaptive CS)')
    plt.grid(True, which="both", ls="-", alpha=0.5)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()

def visualize_patterns(patterns, N, num_patterns=5, save_path=None):
    """
    Visualize the first few measurement patterns.
    
    Args:
        patterns (list): List of pattern vectors.
        N (int): Signal dimension (must be square number for image visualization).
        num_patterns (int): Number of patterns to show.
        save_path (str, optional): Path to save the plot.
    """
    side = int(np.sqrt(N))
    if side * side != N:
        print(f"Signal dimension {N} is not a perfect square. Skipping pattern visualization.")
        return
        
    num_to_show = min(num_patterns, len(patterns))
    
    plt.figure(figsize=(3 * num_to_show, 3))
    
    for i in range(num_to_show):
        plt.subplot(1, num_to_show, i + 1)
        pattern_img = patterns[i].reshape(side, side)
        plt.imshow(pattern_img, cmap='gray')
        plt.title(f'Pattern {i+1}')
        plt.axis('off')
        
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()
