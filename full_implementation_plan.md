# Full Project Implementation Plan

## 1. Modular Architecture
We will split the monolithic script into reusable modules:
*   `src/transforms.py`: Handles forward/inverse transforms (Wavelet, DCT, Hadamard).
*   `src/estimator.py`: The Diagonal Kalman Filter.
*   `src/strategy.py`: Defines the "Next Measurement" policy (Random, Low-Freq, Adaptive).
*   `src/experiment.py`: Runs a simulation loop.
*   `src/benchmark.py`: Runs multiple experiments and generates plots.

## 2. Transforms
*   **Wavelet**: Use `pywt`.
*   **DCT**: Use `scipy.fft.dctn` (Type II).
*   **Hadamard**: Implement **Fast Walsh-Hadamard Transform (FWHT)**.
    *   Since `scipy` lacks a direct FWHT, we will implement a recursive or iterative version ($O(N \log N)$).
    *   We will support 2D Hadamard by applying 1D FWHT on rows then columns.

## 3. Strategies
*   **Random**: Pick unmeasured coefficients uniformly at random.
*   **ZigZag (Low-Freq)**: For DCT/Hadamard, pick in fixed Zig-Zag order (Low to High freq).
*   **Adaptive (Tree)**: For Wavelets, measure parents -> if significant -> measure children.
*   **Adaptive (Oracle)**: (For benchmarking upper bound) Measure largest true coefficients.

## 4. Benchmarking
*   **Comparisons**:
    1.  Wavelet (Random vs. Adaptive)
    2.  DCT (Random vs. Low-Freq)
    3.  Hadamard (Random vs. Low-Freq)
*   **Metrics**: PSNR vs. Sampling Rate (1% to 20%).

## 5. Output
*   `results/`: Directory to save plots and JSON logs.
