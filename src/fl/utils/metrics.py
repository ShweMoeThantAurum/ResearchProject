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


def mape(pred, target, eps=1e-7, threshold=0.05):
    """Mean Absolute Percentage Error with masking for near-zero targets."""
    # mask out values close to zero (normalised scale)
    mask = target > threshold

    if mask.sum() == 0:
        return float('nan')

    pred_masked = pred[mask]
    target_masked = target[mask]

    return torch.mean(
        torch.abs((pred_masked - target_masked) / (target_masked + eps))
    ).item()
