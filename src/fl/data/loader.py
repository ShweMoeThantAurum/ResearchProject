"""
Data loading utilities for global and per-client datasets.
Handles reading processed tensors and preparing batches for GRU models.
"""

import os
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_tensor(path):
    """Loads a PyTorch tensor saved with torch.save."""
    if not os.path.exists(path):
        raise FileNotFoundError("Tensor file not found: " + path)
    return torch.load(path)


def load_global_splits(base_dir):
    """Loads train, val, test splits from datasets/processed/<dataset>/global/."""
    d = os.path.join(base_dir, "global")
    train = load_tensor(os.path.join(d, "train.pt"))
    val = load_tensor(os.path.join(d, "val.pt"))
    test = load_tensor(os.path.join(d, "test.pt"))
    return train, val, test


def load_client_data(base_dir, role):
    """Loads a single client partition such as roadside.pt."""
    path = os.path.join(base_dir, role + ".pt")
    return load_tensor(path)


def make_loader(X, y, batch_size, shuffle=False):
    """Wraps tensors into a DataLoader."""
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def load_test_loader_for_server(dataset, batch_size=64):
    """
    Loads the server-side test split for final evaluation.
    Returns torch.utils.data.DataLoader
    """
    base = os.path.join("datasets/processed", dataset, "global")
    test = torch.load(os.path.join(base, "test.pt"))
    X_test, y_test = test
    return make_loader(X_test, y_test, batch_size=batch_size, shuffle=False)

def load_client_partition(dataset, role):
    """
    Backwards-compatible interface expected by client_main.py.
    Loads datasets/processed/<dataset>/<role>.pt
    """
    base = os.path.join("datasets/processed", dataset)
    return load_client_data(base, role)
