"""
Dataset loader and PyTorch wrapper for traffic prediction.

Used by both clients (local training) and server (final evaluation).
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TrafficDataset(Dataset):
    """
    PyTorch-compatible dataset wrapping preprocessed sliding window data.

    Attributes:
        X: numpy input sequences [N, seq_len, num_nodes]
        y: numpy targets [N, num_nodes]
    """

    def __init__(self, X, y):
        """Store already-preprocessed arrays."""
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        """Return dataset size."""
        return len(self.X)

    def __getitem__(self, idx):
        """Return a single (input, target) pair."""
        return self.X[idx], self.y[idx]


def load_dataset(proc_dir, client_role=None):
    """
    Load preprocessed data for either:
        - global evaluation (client_role=None)
        - a specific client partition (client_role="vehicle", etc.)

    Args:
        proc_dir: path to processed dataset directory
        client_role: optional client name

    Returns:
        X, y numpy arrays
    """
    if client_role is None:
        X = np.load(os.path.join(proc_dir, "X.npy"))
        y = np.load(os.path.join(proc_dir, "y.npy"))
        return X, y

    # FL client-specific partition
    X = np.load(os.path.join(proc_dir, client_role, "X.npy"))
    y = np.load(os.path.join(proc_dir, client_role, "y.npy"))
    return X, y
