### **1. Problem Formulation: The Light Transport System**

Let the scene be modeled as a linear system relating the input (projector) and output (camera).

- Let $p \in \mathbb{R}^{N}$ be the projector image vector (where $N = W_p \times H_p$).
- Let $c \in \mathbb{R}^{M}$ be the camera image vector (where $M = W_c \times H_c$).
- Let $T \in \mathbb{R}^{M \times N}$ be the **Light Transport Matrix**.

The forward light transport is governed by the linear equation:
$$c = T p$$

**The Objective:**
In Dual Photography, we wish to synthesize an image from the perspective of the projector, effectively determining the transport of light from camera pixels back to projector pixels. By Helmholtz Reciprocity, the Dual Image $p''$ formed by a virtual light source $c''$ at the camera is given by:
$$p'' = T^T c''$$

To achieve this, we must acquire the matrix $T$ (or its non-zero elements).

---

### **2. The Naive Algorithm: Brute Force Scan**

This approach treats the system as a "Black Box" and performs impulse response testing on every input channel.

**Algorithm 1: Impulse Response Acquisition**

1.  **Input:** Projector resolution $N$, Camera resolution $M$.
2.  **Procedure:**
    For each pixel $j \in \{1, \dots, N\}$ in the projector:
    a. Construct input vector $p_j$ such that the $j$-th element is $1$ and all others are $0$ (Standard Basis vector).
    b. Project $p_j$ onto the physical scene.
    c. Capture camera measurement $c_j = T p_j$.
    d. Store $c_j$ as the $j$-th column of matrix $T$.
3.  **Complexity:** Requires $N$ measurements. For a standard $1024 \times 768$ projector, $N \approx 7.8 \times 10^5$ images.
4.  **Status:** **Infeasible** due to time constraints.

---

### **3. The Modified Algorithm (Used in Video)**

_Method: Open-Loop Multiplexing via Gray Codes_

To reduce acquisition time, we utilize optical multiplexing. Instead of sending impulse functions, we send broad spectrum patterns encoded via Gray Codes to minimize decoding errors at bit boundaries.

**Algorithm 2: Logarithmic Basis Scan**

1.  **Basis Decomposition:**
    Let $B = \lceil \log_2 N \rceil$. The projector coordinate space is decomposed into $B$ bit-planes.
2.  **Measurement Loop:**
    For $k = 1$ to $B$:
    a. Generate pattern $u_k$ where pixel $j$ is ON if the $k$-th bit of $\text{GrayCode}(j)$ is 1.
    b. Project $u_k$ and capture $y_k$.
    c. Project inverse pattern $\bar{u}_k$ and capture $\bar{y}_k$ (for robust binarization).
3.  **Reconstruction (Decoding):**
    For each camera pixel $i$:
    a. Construct the binary sequence $S_i = [s_{i,1}, s_{i,2}, \dots, s_{i,B}]$ based on intensity thresholds of $y$.
    b. Decode $S_i$ from Gray to Decimal to find the corresponding projector index $j$.
    c. Update $T_{ij} = 1$.
4.  **Complexity:** $O(\log N)$. Requires $\approx 20-40$ images.
5.  **Control Classification:** **Open-Loop Control**. The sequence of inputs $\{u_k\}$ is predetermined and does not change based on scene feedback.

---

### **4. The Optimal Control Perspective**

In the context of Applied Optimal Control, we reframe the acquisition of $T$ not as "taking pictures," but as a **System Identification** problem under resource constraints.

- **The Plant:** The physical scene (geometry + reflectance).
- **State:** The knowledge of the Transport Matrix $T$. Initially, uncertainty is maximal.
- **Control Input ($u_k$):** The illumination pattern at step $k$.
- **Feedback ($y_k$):** The energy observed by the camera.
- **Observability Constraint:** The matrix $T$ is typically **sparse**. Most projector pixels hit empty space or occluded regions (returning zero energy).

The "Gray Code" method is suboptimal because it blindly scans empty space with high-frequency patterns, expending "control energy" (measurement steps) on states that have zero observability.

---

### **5. The Problem Formulation in Optimal Control**

We frame this as a **Sequential Experiment Design** problem.

**Minimize the Cost Function:**
$$J = \sum_{k=1}^{K} \mathcal{C}(u_k) + \lambda \cdot \mathcal{E}(T_{est})$$

Where:

- $K$ is the number of projection steps (to be minimized).
- $\mathcal{C}(u_k)$ is the cost of a measurement (time).
- $\mathcal{E}(T_{est})$ is the reconstruction error of the Transport Matrix.

**Subject to Dynamics:**
$$y_k = T u_k + \eta$$
(Where $\eta$ is sensor noise).

**Optimal Policy:**
We seek a control policy $\pi(y_{1:k-1})$ that selects the next input $u_k$ to maximize the **Information Gain** regarding $T$.
Since $T$ is sparse, the optimal policy is a **Hierarchical Search (Feedback Control)**. If a coarse region returns no energy, the probability of any sub-pixel in that region being active is 0. Therefore, the optimal control for that branch is $u = 0$ (stop scanning).

---

### **6. The Proposed Solution: Adaptive Subdivision**

We propose a **Closed-Loop Feedback Controller**. The controller projects a coarse block. The camera acts as the observer. If the observer detects energy (residue $> \epsilon$), the controller refines the input bandwidth (subdivides). If no energy is detected, the controller prunes the trajectory.

---

### **7. The Modified Algorithm with Optimal Control**

This algorithm implements the closed-loop policy described above.

**Algorithm 3: Adaptive Hierarchical Feedback Scan**

**Initialization:**

- $Q$: A FIFO queue of regions to scan.
- Push initial state: $Q \leftarrow \{ [0, 0, W_p, H_p] \}$ (The entire projector).

**Control Loop:**
While $Q$ is not empty:

1.  **State Estimation:**
    Pop region $R$ from $Q$.
2.  **Control Calculation ($u_k$):**
    Construct pattern $u_k$ such that pixels inside $R$ are $1$, and outside are $0$.
3.  **Actuation & Measurement (Plant):**
    Project $u_k$.
    Capture camera image $y_k$.
4.  **Feedback Analysis (The "Controller"):**
    Calculate Total Energy $E = \sum_{i=1}^{M} y_k(i)$.

    - **Case A (Pruning - Optimal Decision):**
      If $E < \text{NoiseThreshold}$:
      $\quad$ Discard $R$. (The system is unobservable in this region; do not expend further cost).
    - **Case B (Refinement):**
      If $E \geq \text{NoiseThreshold}$ AND Area($R$) $> 1$:
      $\quad$ Subdivide $R$ into 4 quadrants: $R_1, R_2, R_3, R_4$.
      $\quad$ Push $\{R_1, R_2, R_3, R_4\}$ into $Q$.
    - **Case C (Terminal State):**
      If $E \geq \text{NoiseThreshold}$ AND Area($R$) $== 1$:
      $\quad$ We have identified a non-zero element of $T$.
      $\quad$ Record mapping: Projector pixel $R \to$ Camera pixels $\{i \mid y_k(i) > 0\}$.

**Result:**
This method converges to $T$ with $K$ measurements proportional to the _visible surface area_ of the object, rather than the _resolution of the projector_. For sparse scenes, $K_{adaptive} \ll K_{gray\_code}$.
