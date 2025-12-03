"""
Differential Privacy noise addition for client updates.
"""

import torch


def apply_dp_noise(update_dict, sigma=0.0):
    """
    Adds Gaussian noise to each tensor in the update.

    Args:
        update_dict: dict of parameter_name -> torch.Tensor
        sigma: std dev of Gaussian noise

    Returns:
        dict with same keys, noise-added tensors
    """
    if sigma <= 0:
        return update_dict

    noisy = {}
    for k, v in update_dict.items():
        noise = torch.randn_like(v) * sigma
        noisy[k] = v + noise

    return noisy
