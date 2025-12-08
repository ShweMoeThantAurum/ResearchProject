"""
Final evaluation of the global model using the test dataset.

Responsibilities:
 - load X_test and y_test (and optionally X_valid / y_valid),
 - run the GRU model,
 - compute MAE, RMSE, and MAPE metrics.

Also provides a validation-evaluation helper used to log convergence
curves (e.g. MAE vs round) during training.
"""

import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.models.simple_gru import SimpleGRU
from src.utils.metrics import mae, rmse, mape


def _load_split_dataset(proc_dir, split="test"):
    """
    Load (X_split, y_split) as a DataLoader.

    Args:
        proc_dir (str): path to processed dataset directory.
        split (str): one of {"train", "valid", "test"}.

    Returns:
        torch.utils.data.DataLoader
    """
    X_path = os.path.join(proc_dir, f"X_{split}.npy")
    y_path = os.path.join(proc_dir, f"y_{split}.npy")

    if not (os.path.exists(X_path) and os.path.exists(y_path)):
        raise FileNotFoundError(f"Missing X_{split} or y_{split} in {proc_dir}")

    X = np.load(X_path)
    Y = np.load(y_path)

    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
    return DataLoader(ds, batch_size=64, shuffle=False)


def _evaluate_on_split(global_state, proc_dir, num_nodes, hidden_size, split="test"):
    """
    Generic helper to evaluate the global model on a given split.

    Args:
        global_state: model state_dict.
        proc_dir: processed dataset directory.
        num_nodes: number of graph nodes.
        hidden_size: GRU hidden dimension.
        split: which split to evaluate on ("train", "valid", "test").
    """
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size).cpu()
    model.load_state_dict(global_state)
    model.eval()

    loader = _load_split_dataset(proc_dir, split=split)

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


def evaluate_final_model(global_state, proc_dir, num_nodes, hidden_size):
    """
    Evaluate the final global model on the TEST set and return metrics.

    This is called once after all federated rounds complete.
    """
    return _evaluate_on_split(
        global_state, proc_dir, num_nodes, hidden_size, split="test"
    )


def evaluate_validation_model(global_state, proc_dir, num_nodes, hidden_size):
    """
    Evaluate the current global model on the VALIDATION set.

    Used inside the training loop to log per-round convergence metrics
    (e.g. MAE vs round) without touching the test set.
    """
    return _evaluate_on_split(
        global_state, proc_dir, num_nodes, hidden_size, split="valid"
    )
