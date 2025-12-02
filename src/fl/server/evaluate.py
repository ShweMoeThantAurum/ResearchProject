"""
Final evaluation of the global model using the test dataset.

Loads X_test and y_test, runs the GRU model, and computes final accuracy
metrics (MAE, RMSE, MAPE).
"""

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.models.simple_gru import SimpleGRU
from src.utils.metrics import mae, rmse, mape


def load_test_dataset(proc_dir):
    """Load X_test.npy and y_test.npy as a DataLoader."""
    X_path = os.path.join(proc_dir, "X_test.npy")
    y_path = os.path.join(proc_dir, "y_test.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Missing X_test or y_test in {proc_dir}")

    X = np.load(X_path)
    Y = np.load(y_path)

    ds = TensorDataset(
        torch.from_numpy(X).float(),
        torch.from_numpy(Y).float()
    )
    return DataLoader(ds, batch_size=64, shuffle=False)


def evaluate_final_model(global_state, proc_dir, num_nodes, hidden_size):
    """Evaluate the final global model on the test set and return metrics."""
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size).cpu()
    model.load_state_dict(global_state)
    model.eval()

    loader = load_test_dataset(proc_dir)

    preds = []
    trues = []

    with torch.no_grad():
        for X, y in loader:
            out = model(X)
            preds.append(out.numpy())
            trues.append(y.numpy())

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    preds_t = torch.from_numpy(preds)
    trues_t = torch.from_numpy(trues)

    return {
        "MAE": mae(preds_t, trues_t),
        "RMSE": rmse(preds_t, trues_t),
        "MAPE": mape(preds_t, trues_t),
    }
