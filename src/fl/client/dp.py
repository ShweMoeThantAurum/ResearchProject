"""
Differential Privacy: Gaussian noise addition.

Enabled if DP_ENABLED=true.
"""

import os
import torch
import numpy as np


def apply_dp_noise(update):
    """Add independent Gaussian noise to each parameter tensor."""
    sigma = float(os.environ.get("DP_SIGMA", 0.05))

    noisy_update = {}
    for k, v in update.items():
        arr = v.numpy()
        noise = np.random.normal(0, sigma, size=arr.shape).astype(arr.dtype)
        noisy_update[k] = torch.tensor(arr + noise)
    return noisy_update
