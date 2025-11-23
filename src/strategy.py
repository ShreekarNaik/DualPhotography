import numpy as np
import heapq

class StrategyBase:
    def __init__(self, n_coeffs, shape):
        self.n_coeffs = n_coeffs
        self.shape = shape
        self.measured_indices = set()

    def next_index(self, current_estimate=None):
        raise NotImplementedError
    
    def get_name(self):
        raise NotImplementedError

class RandomStrategy(StrategyBase):
    def __init__(self, n_coeffs, shape):
        super().__init__(n_coeffs, shape)
        self.indices = np.arange(n_coeffs)
        np.random.shuffle(self.indices)
        self.pointer = 0
        
    def next_index(self, current_estimate=None):
        if self.pointer >= self.n_coeffs:
            return None
        idx = self.indices[self.pointer]
        self.pointer += 1
        self.measured_indices.add(idx)
        return idx
    
    def get_name(self):
        return "Random"

class LowFreqStrategy(StrategyBase):
    """
    Scans coefficients in a Zig-Zag order (Low Freq -> High Freq).
    Assumes 2D structure flattened.
    """
    def __init__(self, n_coeffs, shape):
        super().__init__(n_coeffs, shape)
        self.indices = self._zigzag_indices(shape[0], shape[1])
        self.pointer = 0
        
    def _zigzag_indices(self, rows, cols):
        # Generate zig-zag indices for a matrix of size rows x cols
        # This is a standard JPEG ZigZag pattern generator
        indices = []
        for s in range(rows + cols - 1):
            if s % 2 == 0:
                # Upwards
                r = s if s < rows else rows - 1
                c = 0 if s < rows else s - rows + 1
                while r >= 0 and c < cols:
                    indices.append(r * cols + c)
                    r -= 1
                    c += 1
            else:
                # Downwards
                r = 0 if s < cols else s - cols + 1
                c = s if s < cols else cols - 1
                while r < rows and c >= 0:
                    indices.append(r * cols + c)
                    r += 1
                    c -= 1
        return indices

    def next_index(self, current_estimate=None):
        if self.pointer >= len(self.indices):
            return None
        idx = self.indices[self.pointer]
        self.pointer += 1
        self.measured_indices.add(idx)
        return idx

    def get_name(self):
        return "Low-Freq (ZigZag)"

class AdaptiveOracleStrategy(StrategyBase):
    """
    Cheating strategy: Measures the largest TRUE coefficients first.
    Used as an upper bound for performance.
    """
    def __init__(self, n_coeffs, shape, true_coeffs):
        super().__init__(n_coeffs, shape)
        # Sort indices by magnitude of true coefficients
        self.indices = np.argsort(np.abs(true_coeffs))[::-1]
        self.pointer = 0
        
    def next_index(self, current_estimate=None):
        if self.pointer >= self.n_coeffs:
            return None
        idx = self.indices[self.pointer]
        self.pointer += 1
        self.measured_indices.add(idx)
        return idx

    def get_name(self):
        return "Adaptive (Oracle)"
