"""
Utilities Module

Provides visualization, metrics, and signal processing utilities.
"""

from .visualization import plot_convergence, plot_uncertainty, visualize_patterns
from .metrics import reconstruction_error, measurement_efficiency
from .signal_processing import generate_sparse_signal, get_sparsifying_basis

__all__ = [
    'plot_convergence',
    'plot_uncertainty',
    'visualize_patterns',
    'reconstruction_error',
    'measurement_efficiency',
    'generate_sparse_signal',
    'get_sparsifying_basis'
]
