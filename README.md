# Adaptive Dual Photography via Optimal Control

**Course**: Applied Optimal Control (AE662)  
**Topic**: Dual Photography using Adaptive Compressed Sensing  
**Implementation**: Python with NumPy/SciPy

[![Tests](https://img.shields.io/badge/tests-passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.8+-blue)]()

## Overview

This project implements **adaptive compressed sensing** for dual photography using optimal control theory. The algorithm adaptively selects measurement patterns to minimize the number of required measurements while maintaining reconstruction quality.

### Key Results
- ✅ **1.98x improvement** over random compressed sensing
- ✅ Greedy information maximization via eigendecomposition
- ✅ Kalman filter-based state estimation
- ✅ Complete test suite with 100% pass rate

## Quick Start

### Installation

```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project (install dependencies)
./run.sh setup
```

### Running the MVP Test

```bash
# Run with default parameters
./run.sh test-mvp

# Run with custom parameters
./run.sh test-mvp --N 100 --sparsity 10 --measurements 60 --noise 0.01 --output results

# View results
open results/convergence.png
open results/uncertainty.png
open results/patterns.png
```

### Running Tests

```bash
# Run all tests
./run.sh test

# Or use pytest directly
uv run pytest src/tests/ -v
```

## Project Structure

```
src/
├── optimal_control/          # Core optimal control algorithms
│   ├── state_estimator.py    # Kalman filter state estimation
│   └── pattern_selector.py   # Eigendecomposition-based pattern selection
├── compressed_sensing/        # Compressed sensing algorithms
│   ├── adaptive_cs.py        # Main adaptive CS algorithm
│   └── random_cs.py          # Baseline: random measurement CS
├── environment/               # Measurement environments
│   ├── synthetic.py          # MVP: Synthetic test environment
│   └── blender_env.py        # Blender integration (stub)
├── utils/                     # Utilities
│   ├── visualization.py      # Plotting and visualization
│   ├── metrics.py            # Performance metrics
│   └── signal_processing.py  # Signal generation and basis functions
├── tests/                     # Test suite
│   ├── test_optimal_control.py
│   ├── test_compressed_sensing.py
│   └── test_integration.py
└── main_mvp.py               # Main entry point for MVP test
```

## Algorithm Overview

The adaptive compressed sensing algorithm implements:

1. **State Estimation** (Kalman Filter):
   - State: `θ_hat` (signal estimate) and `Σ` (covariance)
   - Update equations based on new measurements

2. **Optimal Control** (Greedy Pattern Selection):
   - Select pattern `u_k` as max eigenvector of `Σ_k`
   - Maximizes expected information gain

3. **Measurement Loop**:
   ```
   while uncertainty > threshold:
       u_k = argmax eigenvalue(Σ_k)
       y_k = measure(u_k)
       update_state(y_k, u_k)
   ```

See [`project_guide.md`](project_guide.md) for detailed mathematical formulation.

## Usage Examples

### Basic Adaptive CS

```python
from environment.synthetic import SyntheticEnvironment
from compressed_sensing.adaptive_cs import AdaptiveCS

# Create environment
env = SyntheticEnvironment(N=100, k_sparsity=10, sigma_noise=0.01)
theta_true = env.get_true_signal()

# Run adaptive CS
adaptive_cs = AdaptiveCS(N=100, sigma_noise=0.01)
theta_hat, history = adaptive_cs.run(env, K_measurements=50, theta_true=theta_true)

# Check error
error = np.linalg.norm(theta_hat - theta_true)
print(f"Reconstruction error: {error:.4f}")
```

### Visualization

```python
from utils.visualization import plot_convergence, plot_uncertainty

# Plot convergence comparison
plot_convergence(history_adaptive, history_random, save_path='convergence.png')

# Plot uncertainty reduction
plot_uncertainty(history_adaptive, save_path='uncertainty.png')
```

## Blender Integration (Future)

Blender integration is implemented as a stub in `src/environment/blender_env.py`. To use:

```bash
# Run Blender with Python script (when implemented)
./run.sh run-blender --scene simple
```

**Note**: Blender integration requires:
- Blender 3.0+ installed
- Python bpy module (comes with Blender)
- Additional setup for projector texture nodes

## Mathematical Background

This project applies concepts from Applied Optimal Control:

- **Pontryagin's Maximum Principle**: Optimal control policy derivation
- **Hamilton-Jacobi-Bellman Equation**: Value function and dynamic programming
- **Kalman Filtering**: Recursive state estimation
- **Information Theory**: Mutual information maximization

Key insight: The optimal measurement pattern is the **eigenvector corresponding to the largest eigenvalue** of the uncertainty covariance matrix.

## Testing

The project includes comprehensive tests:

- **Unit Tests**: Individual component correctness (Kalman filter, eigendecomposition)
- **Integration Tests**: End-to-end pipeline validation
- **Performance Tests**: Comparison with random CS baseline

All tests pass:
```
7 passed in 0.11s
```

## Results

### MVP Test Results (N=100, k=10, K=60)

| Method | Final Error | Improvement |
|--------|------------|-------------|
| **Adaptive CS** | 0.807 | **1.98x** |
| Random CS | 1.600 | baseline |

Generated plots:
- `results/convergence.png` - Error convergence over measurements
- `results/uncertainty.png` - Trace of covariance matrix
- `results/patterns.png` - Visualization of selected patterns

## Dependencies

Managed by `uv`:
- `numpy` - Numerical computations
- `scipy` - Eigendecomposition, signal processing
- `matplotlib` - Visualization
- `scikit-image` - Image processing utilities
- `pytest` - Testing framework

## References

1. **Project Guide**: [`project_guide.md`](project_guide.md) - Detailed mathematical formulation
2. **Course Materials**: AE662 Applied Optimal Control
3. Sen, P., et al. "Dual Photography." *ACM SIGGRAPH 2005*
4. Candès, E., et al. "Robust Uncertainty Principles." *IEEE Trans. IT 2006*

## Contributing

This is a course project. For questions or issues, please refer to the project guide or course materials.

## License

Course Project - AE662 Applied Optimal Control

## Acknowledgments

- Project guide based on optimal control theory concepts
- Synthetic testing framework for MVP validation
- Future Blender integration for realistic simulation
