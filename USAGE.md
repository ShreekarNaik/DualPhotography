# Usage Guide

## Getting Started (2 minutes)

```bash
# Step 1: Setup
./run.sh setup

# Step 2: Run MVP test
./run.sh test-mvp

# Step 3: Check results
ls results/
# You should see: convergence.png, uncertainty.png, patterns.png
```

## Common Use Cases

### 1. Quick Test with Default Settings
```bash
./run.sh test-mvp
```

### 2. Custom Parameters
```bash
./run.sh test-mvp --N 200 --sparsity 15 --measurements 80 --noise 0.005
```

### 3. Run All Examples
```bash
./run.sh examples
```

### 4. Run Test Suite
```bash
./run.sh test
```

## Understanding The Output

After running `./run.sh test-mvp`, you'll see:

```
Running MVP Test with N=100, Sparsity=10, K=60
Running Adaptive Compressed Sensing...
Adaptive CS Final Error: 0.807025
Running Random Compressed Sensing (Baseline)...
Random CS Final Error: 1.600362
```

**What this means:**
- Adaptive CS achieved error of 0.807
- Random CS achieved error of 1.600
- Improvement: 1.98x (adaptive is ~2x better)

## Viewing Results

Three plots are generated in `results/`:

1. **convergence.png** - Shows how error decreases with measurements
   - Solid line = Adaptive CS
   - Dashed line = Random CS
   - Lower is better

2. **uncertainty.png** - Shows how uncertainty (trace of covariance) decreases
   - Should decrease monotonically
   - Demonstrates the Kalman filter working

3. **patterns.png** - Shows the first 5 measurement patterns
   - Visualizes what patterns the algorithm selects
   - Early patterns are often smooth (low frequency)
   - Later patterns are more refined (high frequency)

## Parameter Guide

| Parameter | Symbol | Typical Values | Description |
|-----------|--------|----------------|-------------|
| `--N` | N | 64, 100, 256 | Signal dimension (use square numbers) |
| `--sparsity` | k | 5, 10, 20 | Number of non-zero elements |
| `--measurements` | K | 30, 50, 100 | Number of measurements to take |
| `--noise` | σ | 0.001-0.1 | Noise standard deviation |

**Rules of Thumb:**
- K should be > 2k for good recovery
- Keep k < N/10
- Lower noise = better recovery

## Examples Script

The `examples.py` script demonstrates 4 scenarios:

```bash
./run.sh examples
```

**What it shows:**
1. Basic adaptive CS usage
2. Side-by-side comparison with random CS
3. Performance vs. sparsity level
4. Error vs. measurement budget

## Testing

### Run All Tests
```bash
./run.sh test
```

Should show:
```
7 passed in 0.11s
```

### Run Specific Test File
```bash
uv run pytest src/tests/test_optimal_control.py -v
uv run pytest src/tests/test_compressed_sensing.py -v
uv run pytest src/tests/test_integration.py -v
```

## Python API Usage

You can also use the modules directly in Python:

```python
from src.environment.synthetic import SyntheticEnvironment
from src.compressed_sensing.adaptive_cs import AdaptiveCS

# Create environment
env = SyntheticEnvironment(N=100, k_sparsity=10, sigma_noise=0.01)
theta_true = env.get_true_signal()

# Run adaptive CS
adaptive_cs = AdaptiveCS(N=100, sigma_noise=0.01)
theta_hat, history = adaptive_cs.run(env, K_measurements=50, theta_true=theta_true)

# Check results
import numpy as np
error = np.linalg.norm(theta_hat - theta_true)
print(f"Error: {error:.4f}")
```

## Blender Integration (Advanced)

Currently a framework. To use when complete:

```bash
# Make sure Blender is in your PATH
blender --background --python blender_example.py -- --measurements 30 --resolution 64

# Or interactive mode
blender blender_example.py
```

## Troubleshooting

### "ModuleNotFoundError: No module named 'src'"
- Make sure you're running via `./run.sh` or `uv run`
- Path setup is handled automatically

### Tests failing
- Run `./run.sh setup` to reinstall dependencies
- Check Python version: `python --version` (needs 3.8+)

### Plots not appearing
- Check `results/` directory exists
- Try: `mkdir -p results`
- Ensure matplotlib backend is working

### Poor performance (high error)
- Try more measurements: `--measurements 80`
- Reduce noise: `--noise 0.001`
- Check that sparsity k << N

## File Structure Quick Reference

```
Project/
├── run.sh                # <- START HERE (main entry point)
├── examples.py           # Example scenarios
├── config.py            # Parameters
├── README.md            # Full documentation
├── SUMMARY.md           # Quick summary
├── USAGE.md             # This file
└── src/                 # All source code
    ├── optimal_control/ # Kalman filter, pattern selection
    ├── compressed_sensing/ # Adaptive & random CS
    ├── environment/     # Test environments
    ├── utils/          # Visualization, metrics
    ├── tests/          # Test suite
    └── main_mvp.py     # MVP entry point
```

## Next Steps

1. ✅ Run `./run.sh test-mvp` to see it working
2. ✅ Open `results/convergence.png` to see the improvement
3. ✅ Run `./run.sh examples` for more demonstrations
4. ✅ Read [project_guide.md](project_guide.md) for the math
5. ✅ Modify parameters and experiment!

## Getting Help

- Mathematical details: [project_guide.md](project_guide.md)
- Implementation details: [walkthrough.md](.gemini/antigravity/brain/0c533635-19eb-4cf6-b2a4-c9a16050a62c/walkthrough.md)
- Quick reference: [SUMMARY.md](SUMMARY.md)
- Examples: Run `./run.sh examples`

## Performance Tips

For faster execution:
- Use smaller N (e.g., 64 instead of 256)
- Reduce measurement count
- Use identity basis (default) instead of DCT

For better accuracy:
- Increase measurements
- Reduce noise level
- Use appropriate sparsifying basis

## Advanced: Custom Experiments

Edit `config.py` to change defaults, or create custom scripts:

```python
import sys
sys.path.insert(0, 'src')

from environment.synthetic import SyntheticEnvironment
from compressed_sensing.adaptive_cs import AdaptiveCS

# Your custom experiment here...
```

---

**Quick Start**: `./run.sh setup && ./run.sh test-mvp`
