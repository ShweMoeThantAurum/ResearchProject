"""
Common evaluation metrics for FL experiments.
Implements MAE, RMSE, and MAPE for traffic forecasting.
"""

import torch
import torch.nn.functional as F


def mae(pred, target):
    """Mean Absolute Error."""
    return torch.mean(torch.abs(pred - target)).item()


def rmse(pred, target):
    """Root Mean Squared Error."""
    return torch.sqrt(F.mse_loss(pred, target)).item()


def mape(pred, target):
    """Mean Absolute Percentage Error."""
    eps = 1e-7
    return torch.mean(torch.abs((pred - target) / (target + eps))).item()
