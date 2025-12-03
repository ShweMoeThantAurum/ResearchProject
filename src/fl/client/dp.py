"""
Differential privacy utilities for client updates.
Adds simple Gaussian noise to model parameters.
"""

import torch


def apply_dp_noise(update_dict, sigma):
    """Apply Gaussian DP noise to each tensor in an update."""
    if sigma <= 0:
        return update_dict

    noisy = {}
    for name, tensor in update_dict.items():
        noise = torch.randn_like(tensor) * sigma
        noisy[name] = tensor + noise

    return noisy
