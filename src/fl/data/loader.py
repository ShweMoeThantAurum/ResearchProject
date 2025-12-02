"""
Central loader for preprocessed dataset tensors and role-specific partitions.
Used by both server and clients during FL execution.
"""

import os
import torch

from .partition import partition_clients, create_dataloaders, ROLES


def load_preprocessed(dataset):
    """Load a preprocessed dataset tensor."""
    path = os.path.join("datasets", "processed", f"{dataset}.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Preprocessed dataset not found: {path}")
    return torch.load(path, map_location="cpu")


def infer_num_nodes(tensor):
    """Infer the number of nodes from a preprocessed tensor."""
    return tensor.shape[-1]


def load_client_partition(dataset, role, batch_size):
    """Load train/test loaders for a specific role."""
    tensor = load_preprocessed(dataset)
    num_nodes = infer_num_nodes(tensor)

    per_role = partition_clients(tensor, num_nodes)

    if role not in per_role:
        raise ValueError(f"Unknown role: {role}")

    train_loader, test_loader = create_dataloaders(per_role[role], batch_size)
    return train_loader, test_loader, num_nodes


def load_test_loader_for_server(dataset, batch_size):
    """Load global test loader for server-side evaluation."""
    tensor = load_preprocessed(dataset)
    num_nodes = infer_num_nodes(tensor)

    # Use roadside role test set as evaluation baseline
    per_role = partition_clients(tensor, num_nodes)
    _, test_loader = create_dataloaders(per_role["roadside"], batch_size)

    return test_loader, num_nodes
