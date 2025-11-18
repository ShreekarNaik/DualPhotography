# Project Configuration
# This file documents project settings and parameters

## Core Parameters

# Signal Dimension
# Recommended: 64, 100, 256 (for square patterns: 8x8, 10x10, 16x16)
DEFAULT_N = 100

# Sparsity Level
# Number of non-zero elements in true signal
# Rule of thumb: k < N/10 for good recovery
DEFAULT_SPARSITY = 10

# Noise Level
# Standard deviation of measurement noise
# Typical values: 0.001 (low noise) to 0.1 (high noise)
DEFAULT_SIGMA_NOISE = 0.01

# Measurement Budget
# Number of measurements to take
# Compressed sensing requires: K > 2k * log(N/k) typically
DEFAULT_K_MEASUREMENTS = 50

## Blender Settings

# Resolution for projector patterns and camera
BLENDER_RESOLUTION = 64

# Render engine
BLENDER_RENDER_ENGINE = "CYCLES"  # or "BLENDER_EEVEE"

# Render samples (quality vs speed tradeoff)
BLENDER_SAMPLES = 64  # Higher = better quality, slower

## Output Directories

RESULTS_DIR = "results"
PLOTS_DIR = "results/plots"
DATA_DIR = "results/data"

## Algorithm Settings

# Initial covariance scale
INITIAL_SIGMA_SCALE = 1.0

# Convergence threshold (stop when trace < threshold)
CONVERGENCE_THRESHOLD = 0.01

## Visualization

# Figure size for plots
FIGURE_SIZE = (10, 6)

# DPI for saved figures
FIGURE_DPI = 300

# Number of patterns to visualize
NUM_PATTERNS_TO_SHOW = 5
