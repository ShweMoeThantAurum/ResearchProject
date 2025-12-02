"""
Local differential privacy utilities.

Implements Gaussian noise addition to model parameters,
controlled by sigma (noise magnitude).
"""


import torch


def apply_dp(state_dict, sigma):
    """
    Apply Gaussian noise to all model parameters for local DP.

    Parameters:
        state_dict : PyTorch parameter dictionary
        sigma      : noise standard deviation
    """
    for name in state_dict:
        noise = torch.randn_like(state_dict[name]) * sigma
        state_dict[name].add_(noise)
