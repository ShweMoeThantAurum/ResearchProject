"""
Data loading utilities for global and per-client datasets.
Handles reading processed tensors and preparing batches for GRU models.
"""

import os
import torch

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

def load_client_partition(dataset, role):
    """Loads a client's partition from datasets/processed/<dataset>/<role>.pt."""
    base = os.path.join("datasets", "processed", dataset)
    path = os.path.join(base, role + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError("Client partition missing: " + path)
    return torch.load(path)  # (X, y)
