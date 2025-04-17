import numpy as np
import torch
from scipy.ndimage import gaussian_filter


def make_grid(n, L):
    complex_step = complex(0, n)
    x, y, z = np.ogrid[-L:L:complex_step, -L:L:complex_step, -L:L:complex_step]
    return np.meshgrid(x, y, z)

def create_constant_m(X, value=1.0):
    return np.ones_like(X) * value
