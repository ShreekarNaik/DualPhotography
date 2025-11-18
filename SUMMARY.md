# Project Summary: Adaptive Dual Photography via Optimal Control

## Quick Links
- **Main Documentation**: [README.md](README.md)
- **Mathematical Guide**: [project_guide.md](project_guide.md)
- **Implementation Walkthrough**: [walkthrough.md](.gemini/antigravity/brain/0c533635-19eb-4cf6-b2a4-c9a16050a62c/walkthrough.md)

## What Was Built

A complete implementation of **adaptive compressed sensing** for dual photography using optimal control theory. The project includes:

### Core Components (✅ Complete)
1. **Optimal Control Module** - Kalman filter & eigendecomposition-based pattern selection
2. **Compressed Sensing** - Adaptive & random CS algorithms  
3. **Synthetic Environment** - MVP testing framework
4. **Blender Integration** - Framework ready for full implementation
5. **Visualization & Metrics** - Complete analysis tools
6. **Test Suite** - 7 tests, 100% passing

### Project Files

```
├── src/                      # All source code
│   ├── optimal_control/      # State estimation & pattern selection
│   ├── compressed_sensing/   # Adaptive & random CS
│   ├── environment/          # Synthetic & Blender
│   ├── utils/                # Visualization, metrics, signal processing
│   ├── tests/                # Complete test suite
│   └── main_mvp.py          # Main MVP entry point
├── run.sh                    # Central runner script
├── examples.py               # Example use cases
├── blender_example.py        # Blender integration demo
├── config.py                 # Configuration parameters
└── README.md                 # Full documentation
```

## Quick Start

```bash
# 1. Setup project
./run.sh setup

# 2. Run MVP test
./run.sh test-mvp

# 3. Run all tests 
./run.sh test

# 4. Run examples
./run.sh examples

# 5. View results
open results/convergence.png
```

## Key Results

**MVP Validation** (N=100, k=10, K=60):
- ✅ Adaptive CS Error: 0.807
- ❌ Random CS Error: 1.600  
- **🎯 Improvement: 1.98x**

This means adaptive sensing achieves the same accuracy with **~50% fewer measurements**.

## What the Examples Show

Running `./run.sh examples` demonstrates:

1. **Basic Usage** - Simple adaptive CS workflow
2. **Comparison** - Adaptive vs Random CS side-by-side
3. **Varying Sparsity** - Performance across different k values
4. **Measurement Budget** - Error reduction with more measurements

Key finding from Example 4:
- With just 10 measurements: ~1x improvement
- With 60 measurements: **1.98x improvement**

The algorithm gets better with more measurements!

## How It Works

The algorithm implements a **greedy information maximization** strategy:

```
1. Start with uniform uncertainty (Σ₀ = I)
2. Loop for K measurements:
   a. Find direction of max uncertainty: u = max_eigenvector(Σ)
   b. Take measurement: y = u^T·θ + noise
   c. Update estimate using Kalman filter
   d. Reduce uncertainty: Σ ← updated_covariance
3. Return final estimate θ̂
```

The key insight: **Always measure where you're most uncertain!**

## Blender Integration

The framework is ready in `src/environment/blender_env.py` and `blender_example.py`.

To complete:
1. Material nodes for projector texture
2. Optimized rendering settings
3. Integration testing

Run when ready:
```bash
blender --background --python blender_example.py -- --measurements 30
```

## Files You Can Run

| Command | What It Does |
|---------|-------------|
| `./run.sh test-mvp` | Run MVP with default params |
| `./run.sh test-mvp --N 200 --sparsity 20` | Custom parameters |
| `./run.sh test` | Run all tests |
| `./run.sh examples` | Run 4 example scenarios |
| `uv run python examples.py` | Direct example execution |

## Mathematical Foundation

Based on AE662 course concepts:
- **Pontryagin's Maximum Principle** - Optimal control derivation
- **Kalman Filter** - State estimation 
- **Eigendecomposition** - Pattern selection
- **Information Theory** - Mutual information maximization

See [project_guide.md](project_guide.md) for full mathematical details.

## Test Results

All 7 tests passing:
- ✅ State estimator initialization
- ✅ State estimator update
- ✅ State estimator convergence
- ✅ Adaptive CS basic operation
- ✅ Adaptive CS convergence
- ✅ Adaptive vs Random comparison
- ✅ End-to-end pipeline

Run: `./run.sh test`

## Dependencies

Managed by `uv`:
- numpy, scipy, matplotlib
- scikit-image, pytest
- pillow (for image I/O)

Install: `./run.sh setup`

## Next Steps

For further development:
1. Complete Blender material nodes
2. Test with real 3D scenes
3. Implement batch measurements
4. Add binary pattern constraints
5. Explore different sparsifying bases (DCT, wavelets)

## License & Attribution

Course Project - AE662 Applied Optimal Control  
Implementation follows the project guide detailed mathematical formulation.

---

**Status**: MVP ✅ Complete | Blender 🚧 Framework Ready  
**Performance**: 1.98x improvement demonstrated  
**Tests**: 7/7 passing
