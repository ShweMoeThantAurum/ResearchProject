"""
Data loading utilities for global and per-client datasets.
Handles reading processed tensors and preparing batches for GRU models.
"""

import os
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_tensor(path):
    """Load a tensor or (X, y) tuple saved with torch.save."""
    if not os.path.exists(path):
        raise FileNotFoundError("Tensor file not found: " + path)
    return torch.load(path)


def load_global_splits(base_dir):
    """Load train/val/test splits from processed global directory."""
    d = os.path.join(base_dir, "global")
    train = load_tensor(os.path.join(d, "train.pt"))
    val = load_tensor(os.path.join(d, "val.pt"))
    test = load_tensor(os.path.join(d, "test.pt"))
    return train, val, test


def load_client_data(base_dir, role):
    """Load a single client partition file such as roadside.pt."""
    path = os.path.join(base_dir, role + ".pt")
    return load_tensor(path)


def make_loader(X, y, batch_size, shuffle):
    """Wrap tensors into a DataLoader."""
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_test_loader_for_server(dataset, batch_size=64):
    """
    Load the server-side test split for final evaluation.
    """
    base = os.path.join("datasets", "processed", dataset, "global")
    test = torch.load(os.path.join(base, "test.pt"))
    X_test, y_test = test
    return make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)


def load_client_partition(dataset, role):
    """
    Load one client partition as raw tensors (X, y).

    Files:
        datasets/processed/<dataset>/<role>.pt
    """
    base = os.path.join("datasets", "processed", dataset)
    X, y = load_client_data(base, role)
    return X, y

