"""
Differential privacy utilities for model updates.

Provides a simple Gaussian mechanism that adds noise to each parameter
tensor in a state_dict, modelling local DP-style protection.
"""

import torch


def dp_add_noise(state_dict, sigma=0.05):
    """Add Gaussian noise with standard deviation sigma to model parameters."""
    noisy = {}
    for k, v in state_dict.items():
        noise = torch.randn_like(v) * sigma
        noisy[k] = v + noise
    return noisy
