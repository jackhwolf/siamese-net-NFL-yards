from torch import from_numpy
import numpy as np

def to_torch(arr):
    if isinstance(arr, np.ndarray):
        return from_numpy(arr.astype(np.float32))
    return arr

class EarlyStopper:

    def __init__(self, patience=10, tolerance=1e-5):
        self.count = 0
        self.patience = patience
        self.tolerance = tolerance
        self.most_recent_val = None

    def __call__(self, val):
        out = False
        if self.most_recent_val is None:
            out = False
        elif np.abs(val - self.most_recent_val) >= self.tolerance:
            self.count = 0
        else:
            self.count += 1
            out = self.count >= self.patience
        self.most_recent_val = val
        return out            