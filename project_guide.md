# Course Project Guide: Adaptive Compressed Sensing via Optimal Control

## 1. Project Overview
**Title:** Adaptive Compressed Sensing for Dual Photography: An Optimal Control Approach

**Objective:**
To reconstruct a high-resolution scene (e.g., 1024x1024) in a "Dual Photography" setup using the minimum number of measurements. We formulate this as a **Time-Optimal Control Problem** where the "controller" decides which light pattern to project next to maximize information gain, updating its belief about the image in real-time.

**Core Concept:**
Instead of a blind scan (raster) or a fixed random scan (standard Compressed Sensing), we use an **Adaptive Strategy**. We start by measuring the "coarse" structure (low frequencies). Based on what we see, we "steer" our measurements toward the detailed regions (edges, textures), ignoring the empty flat regions. This exploits the **Sparsity** of images in the Wavelet/DCT domain.

---

## 2. Problem Formulation (The Maths)

We frame this as a **Partially Observable System** where we control the measurement process.

### 2.1. The State Space ($x$)
The "State" is not the pixel values directly, but the **Transform Coefficients** (e.g., Wavelet or DCT coefficients) of the image.
*   Let $I$ be the $N \times N$ image (vectorized size $M = N^2$).
*   Let $\Psi$ be a sparsifying transform (Wavelet/DCT).
*   The state vector is $x = \Psi I$.
*   **Assumption:** $x$ is sparse (mostly zeros).

### 2.2. The Measurement Model ($y_k$)
At step $k$, we choose a control input $u_k$ (a measurement pattern).
*   $y_k = u_k^T I + v_k$ (Measurement = Projection of Pattern onto Image + Noise)
*   In terms of state $x$: $y_k = u_k^T (\Psi^{-1} x) + v_k = (h_k)^T x + v_k$
    *   Where $h_k = (\Psi^{-1})^T u_k$ is the effective measurement vector in the transform domain.

### 2.3. The Estimator (The "Observer")
Since $M$ is large ($10^6$), a full Kalman Filter ($P$ matrix of size $10^{12}$) is impossible.
**Solution:** Use a **Diagonal Kalman Filter**. We assume coefficients are independent.
*   **Belief State:**
    *   $\hat{x}_k \in \mathbb{R}^M$: Current estimate of coefficients.
    *   $P_k \in \mathbb{R}^M$: Diagonal variance (uncertainty) of each coefficient.

**Update Equations (Scalar Kalman Filter for each coefficient $i$):**
For the measured coefficient $i$ (if we measure directly in the basis):
$$ K_k = \frac{P_{k,i}}{P_{k,i} + R} $$
$$ \hat{x}_{k+1,i} = \hat{x}_{k,i} + K_k (y_k - \hat{x}_{k,i}) $$
$$ P_{k+1,i} = (1 - K_k) P_{k,i} $$

### 2.4. The Control Problem (The "Policy")
**Goal:** Minimize final uncertainty $\sum P_N$ using $N$ measurements.
**Control Input:** Which basis vector $u_k$ to select next?
**Optimal Policy (Greedy):** Select the index $i$ that has the **highest expected energy** or **highest uncertainty**.
*   In Wavelets, we use the "Parent-Child" property: If a parent coefficient is large, its children are likely large.
*   **Algorithm:**
    1.  Measure all coarse coefficients (Low Frequency).
    2.  Identify "significant" coefficients (above threshold).
    3.  Add their "children" (higher frequency details at same location) to the **Priority Queue**.
    4.  Pick the next measurement from the Priority Queue.

---

## 3. Algorithm Selection & Justification (Why Kalman?)

You asked: *Is Kalman/Bayesian update the best?*

Based on the literature (e.g., *Review of Single-Pixel Imaging*, *Dual Photography*):
1.  **Standard Compressed Sensing (L1 Minimization):**
    *   **Pros:** Mathematically robust, handles correlations well.
    *   **Cons:** "Batch" process. You take measurements *then* solve a huge optimization problem. Slow for 1024x1024 images ($N=10^6$). Hard to make "Adaptive" in real-time.
2.  **Deep Learning (Generative Models):**
    *   **Pros:** State-of-the-art quality.
    *   **Cons:** Requires massive training datasets. "Black box" - less about "Optimal Control theory" and more about "Data approximation".
3.  **Our Approach (Diagonal Kalman Filter + Adaptive Control):**
    *   **Why it fits this course:** It frames the problem as a **Dynamic System**.
    *   **Efficiency:** $O(N)$ complexity. We update our belief *instantly* after every measurement.
    *   **Adaptivity:** Allows us to decide the *next* measurement based on the *current* belief. This is the essence of **Feedback Control**.
    *   **Hadamard vs. Fourier:** As noted in *Hadamard vs Fourier*, Hadamard is better for binary DMDs. Our formulation works with any basis. We use Wavelets here for their tree structure (easy to predict "children"), but the math holds for Hadamard too.

**Verdict:** For a project on *Optimal Control*, the Kalman Filter is the most theoretically sound choice because it is the **Optimal Recursive Estimator**.

---

## 4. Why This Fits "Optimal Control"
*   **Dynamics:** The "Information State" (Estimate + Covariance) evolves deterministically based on our measurements.
*   **Control:** We actively choose $u_k$ to steer the covariance $P_k$ to zero.
*   **Cost Function:** Minimum time (measurements) to reach target error.
*   **Feedback:** The next measurement depends on the current estimate $\hat{x}_k$.

---

## 4. Execution & Simulation Plan

### 4.1. Code Structure (`src/`)
We have implemented a modular simulation framework:
*   **`transforms.py`**: Handles **Wavelet**, **DCT**, and **Hadamard (FWHT)** transforms.
*   **`estimator.py`**: Implements the **Diagonal Kalman Filter**.
*   **`strategy.py`**: Defines the control policies:
    *   `RandomStrategy`: Standard Compressed Sensing baseline.
    *   `LowFreqStrategy`: Zig-Zag scan (effective for DCT/Hadamard).
    *   `AdaptiveOracleStrategy`: Optimal control upper bound (measures largest coefficients).
*   **`benchmark.py`**: Runs a full comparison of all methods.

### 4.2. Running the Project
1.  **Install Dependencies**: `./manage.sh install`
2.  **Run Benchmark**: `./manage.sh benchmark`
    *   This will generate a plot in `results/benchmark_plot.png` comparing PSNR vs. Measurements for all strategies.

### 4.3. Expected Results
*   **Adaptive/Low-Freq** methods should significantly outperform **Random** sampling.
*   **Wavelet Adaptive** is typically best for natural images with sharp edges.
*   **Hadamard** is practical for hardware (DMDs) and performs well with Low-Freq scanning.

---

## 5. Explanatory Corner (ELI5)

### What is "Sparsity"?
Imagine a book. Most pages are empty, only a few have text. If you know *which* pages have text, you can just read those and ignore the rest.
*   **Real Images** are like this book in the "Wavelet Domain". Most coefficients are zero.
*   **The Challenge:** We don't know *which* pages have text beforehand.
*   **Adaptive Strategy:** We read the "Table of Contents" (Low Frequencies) first. It tells us which chapters are interesting. Then we only read those chapters.

### Why not a full Kalman Filter?
A full Kalman Filter tracks how every pixel relates to every other pixel. For a 1MP image, that's a trillion connections. Too big!
**Diagonal KF:** We assume pixels (or coefficients) are independent. It's like treating each page of the book separately. Much faster ($O(N)$), and good enough for this task.

### Dual Photography Connection
In Dual Photography, we project patterns to "see" from the projector's view.
*   **Standard:** Project every single pixel (Millions of patterns). Slow!
*   **Adaptive:** Project broad patterns first. If a region is dark (no light transport), don't bother projecting fine details there. Move to the bright regions. This is exactly what our algorithm does.
