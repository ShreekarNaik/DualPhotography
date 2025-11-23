# Adaptive Compressed Sensing Project

This project implements an **Optimal Control** approach to Compressed Sensing, specifically for **Dual Photography**.
It simulates the reconstruction of an image using different measurement bases (Wavelet, DCT, Hadamard) and sampling strategies (Random, Low-Frequency, Adaptive).

## Structure
*   `src/transforms.py`: Implementations of Wavelet, DCT, and Fast Walsh-Hadamard Transforms.
*   `src/estimator.py`: Diagonal Kalman Filter for recursive estimation.
*   `src/strategy.py`: Measurement selection policies (Random, ZigZag, Oracle).
*   `src/experiment.py`: Simulation loop.
*   `src/benchmark.py`: Main script to run comparisons.

## Usage

### Prerequisites
Install dependencies using `uv` (or pip):
```bash
./manage.sh install
```

### Run Simulation (Single Run)
Runs a basic demo with Wavelets:
```bash
./manage.sh sim
```

### Run Full Benchmark
Runs all combinations (Wavelet/DCT/Hadamard x Random/Adaptive) and generates a plot in `results/`:
```bash
./manage.sh benchmark
```

## Results
Check `results/benchmark_plot.png` to see the PSNR comparison.
*   **Adaptive (Oracle)** should show the best performance (Upper Bound).
*   **Low-Freq (ZigZag)** works well for DCT/Hadamard on natural images.
*   **Random** is the baseline (Standard Compressed Sensing).
