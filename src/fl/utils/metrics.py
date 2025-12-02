"""
Common evaluation metrics for traffic prediction.

Provides MAE, RMSE, and MAPE implemented on top of PyTorch tensors.
"""

import torch


def mae(pred, true):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - true)).item()


def rmse(pred, true):
    """Root Mean Squared Error."""
    return torch.sqrt(torch.mean((pred - true) ** 2)).item()


def mape(pred, true, eps=1e-6):
    """
    Mean Absolute Percentage Error (in %).

    Adds a small epsilon to avoid division by zero.
    """
    denom = torch.clamp(torch.abs(true), min=eps)
    return (torch.mean(torch.abs((pred - true) / denom)) * 100.0).item()
