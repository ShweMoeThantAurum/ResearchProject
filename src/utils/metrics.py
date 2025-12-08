"""
Metric functions for evaluating traffic forecasting models.

Implements MAE, RMSE, MAPE, sMAPE, and R², together with a helper that
evaluates all metrics over a PyTorch DataLoader.
"""

import torch
import torch.nn.functional as F


def mae(yhat, y):
    """Mean absolute error (MAE)."""
    return torch.mean(torch.abs(yhat - y)).item()


def rmse(yhat, y):
    """Root mean squared error (RMSE)."""
    return torch.sqrt(F.mse_loss(yhat, y)).item()


def mape(yhat, y):
    """Mean absolute percentage error with small epsilon for stability."""
    eps = 1e-6
    return torch.mean(torch.abs((yhat - y) / (y + eps))).item()


def smape(yhat, y):
    """Symmetric mean absolute percentage error (sMAPE)."""
    eps = 1e-6
    return torch.mean(
        2.0 * torch.abs(yhat - y) / (torch.abs(yhat) + torch.abs(y) + eps)
    ).item()


def r2_score(yhat, y):
    """Coefficient of determination (R²) for regression."""
    y_mean = torch.mean(y)
    ss_tot = torch.sum((y - y_mean) ** 2)
    ss_res = torch.sum((y - yhat) ** 2)
    return (1 - ss_res / (ss_tot + 1e-8)).item()


@torch.no_grad()
def eval_loader_metrics(model, loader, device):
    """
    Compute all core metrics over a DataLoader using the current model.

    Returns a tuple:
      (MAE, RMSE, MAPE, sMAPE, R²)
    """
    model.eval()
    all_mae, all_rmse, all_mape, all_smape, all_r2 = [], [], [], [], []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yhat = model(x)

        all_mae.append(mae(yhat, y))
        all_rmse.append(rmse(yhat, y))
        all_mape.append(mape(yhat, y))
        all_smape.append(smape(yhat, y))
        all_r2.append(r2_score(yhat, y))

    n = max(1, len(all_mae))

    return (
        sum(all_mae) / n,
        sum(all_rmse) / n,
        sum(all_mape) / n,
        sum(all_smape) / n,
        sum(all_r2) / n,
    )
