"""
Client-side dataset loading and preprocessing utilities.

Each client loads its own (X, y) partition, adjusts shapes to match
the model configuration, and builds a PyTorch DataLoader for training.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.fl.logger import log_event


def client_file_paths(proc_dir, role):
    """
    Construct file paths for this client's partition:

        <proc_dir>/clients/client_<role>_X.npy
        <proc_dir>/clients/client_<role>_y.npy
    """
    root = os.path.join(proc_dir, "clients")
    return (
        os.path.join(root, f"client_{role}_X.npy"),
        os.path.join(root, f"client_{role}_y.npy"),
    )


def pad_or_trim_last_dim(arr, target):
    """
    Ensure the last dimension matches `target`.

    - truncate if too large
    - zero-pad if too small
    """
    *_, d = arr.shape
    if d == target:
        return arr
    if d > target:
        return arr[..., :target]

    pad = [(0, 0)] * arr.ndim
    pad[-1] = (0, target - d)
    return np.pad(arr, pad, mode="constant")


def load_local_data(proc_dir, role, num_nodes, batch_size, local_epochs, lr):
    """
    Load this client's local dataset and return a DataLoader.
    """
    x_path, y_path = client_file_paths(proc_dir, role)

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"[{role}] Missing file: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"[{role}] Missing file: {y_path}")

    X = pad_or_trim_last_dim(np.load(x_path), num_nodes)
    y = pad_or_trim_last_dim(np.load(y_path), num_nodes)

    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    print(f"[{role}] Loaded data: X={X.shape}, y={y.shape}, batch={batch_size}")

    log_event(
        "client_data.log",
        {
            "role": role,
            "X_shape": list(X.shape),
            "y_shape": list(y.shape),
            "batch_size": batch_size,
            "local_epochs": local_epochs,
            "lr": lr,
        },
    )

    return loader
