"""
Standard regression metrics for traffic flow prediction.
"""

import numpy as np


def mae(y_true, y_pred):
    """Compute mean absolute error."""
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    """Compute root mean squared error."""
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred):
    """Compute mean absolute percentage error."""
    eps = 1e-6
    return float(np.mean(np.abs((y_true - y_pred) / (y_true + eps))))
