"""
Partitioning logic for creating role-specific datasets for FL clients.
Splits preprocessed tensors into per-role partitions and train/test loaders.
"""

import os
import torch
from torch.utils.data import TensorDataset, DataLoader


ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def split_by_nodes(tensor, num_nodes):
    """Split sequence windows by nodes dimension."""
    # tensor shape: [N, seq, nodes]
    return tensor[:, :, :num_nodes]


def partition_clients(data_tensor, num_nodes):
    """Slice per-role data partitions from the sequence tensor."""
    per_role = {}

    # Each role gets its own slice of nodes
    nodes_per_role = num_nodes // len(ROLES)

    for i, role in enumerate(ROLES):
        start = i * nodes_per_role
        end = start + nodes_per_role
        sub = data_tensor[:, :, start:end]
        per_role[role] = sub

    return per_role


def create_dataloaders(role_tensor, batch_size):
    """Build train and test loaders for a role."""
    X = role_tensor[:, :-1, :]   # all but last step
    Y = role_tensor[:, -1, :]    # last step is prediction target

    size = len(X)
    split = int(size * 0.8)

    X_train, Y_train = X[:split], Y[:split]
    X_test, Y_test = X[split:], Y[split:]

    train_ds = TensorDataset(X_train, Y_train)
    test_ds = TensorDataset(X_test, Y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
