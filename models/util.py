from torch import from_numpy
import numpy as np

def to_torch(arr):
    if isinstance(arr, np.ndarray):
        return from_numpy(arr.astype(np.float32))
    return arr