"""
Final evaluation of the global model on the test split.

This module loads X_test and y_test for the chosen dataset, runs
the GRU model, and computes MAE, RMSE and MAPE metrics.
"""

import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.fl.models.gru_model import SimpleGRU
from src.fl.utils.metrics import mae, rmse, mape


def load_test_loader(proc_dir, batch_size):
    """
    Load X_test and y_test from the processed directory and
    wrap them in a DataLoader.
    """
    x_path = os.path.join(proc_dir, "X_test.npy")
    y_path = os.path.join(proc_dir, "y_test.npy")

    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            "Missing test arrays in " + proc_dir
        )

    X = np.load(x_path)
    Y = np.load(y_path)

    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float(),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    return loader


def evaluate_global_model(global_state, proc_dir, num_nodes, hidden_size):
    """
    Evaluate the final global model on the test split.

    Returns a dictionary of metrics:
      {
        "MAE": float,
        "RMSE": float,
        "MAPE": float
      }
    """
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size)
    model.load_state_dict(global_state)
    model.eval()

    loader = load_test_loader(proc_dir, batch_size=64)

    preds = []
    trues = []

    with torch.no_grad():
        for X, Y in loader:
            out = model(X)
            preds.append(out.numpy())
            trues.append(Y.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    preds_t = torch.from_numpy(preds)
    trues_t = torch.from_numpy(trues)

    return {
        "MAE": float(mae(preds_t, trues_t)),
        "RMSE": float(rmse(preds_t, trues_t)),
        "MAPE": float(mape(preds_t, trues_t)),
    }
