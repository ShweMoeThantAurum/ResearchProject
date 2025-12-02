"""
Client-side dataset loading utilities.

Handles loading local X/y partitions, verifying shapes, applying padding
or trimming if needed, and constructing DataLoaders for training.
"""

import os
import numpy as np
import torch
from typing import Tuple
from torch.utils.data import DataLoader, TensorDataset
from src.fl.logger import log_event


def client_file_paths(proc_dir, role):
    """Return the expected X/y file paths for the given client role."""
    root = os.path.join(proc_dir, "clients")
    x_path = os.path.join(root, f"client_{role}_X.npy")
    y_path = os.path.join(root, f"client_{role}_y.npy")
    return x_path, y_path


def pad_or_trim_last_dim(arr, target):
    """
    Ensure array shape matches expected number of nodes.

    Pads with zeros if too small; truncates if too large.
    """
    *_, d = arr.shape
    if d == target:
        return arr
    if d > target:
        return arr[..., :target]

    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, target - d)
    return np.pad(arr, pad_width, mode="constant")


def load_local_data(proc_dir, role, num_nodes, batch_size, local_epochs, lr):
    """Load the client’s local partition and return a DataLoader."""
    x_path, y_path = client_file_paths(proc_dir, role)

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"[{role}] Missing X file: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"[{role}] Missing y file: {y_path}")

    X = pad_or_trim_last_dim(np.load(x_path), num_nodes)
    y = pad_or_trim_last_dim(np.load(y_path), num_nodes)

    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    print(f"[{role}] Loaded local data: X={X.shape}, y={y.shape}, batch={batch_size}")

    log_event("client_data.log", {
        "role": role,
        "X_shape": list(X.shape),
        "y_shape": list(y.shape),
        "batch_size": batch_size,
        "local_epochs": local_epochs,
        "lr": lr,
    })

    return loader
