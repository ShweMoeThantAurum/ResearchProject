"""
Partitioning utilities for assigning dataset shards to FL clients.

Implements simple non-IID partitioning by assigning different subsets
of sensors (nodes) to each client.
"""

import os
import numpy as np
from src.fl.utils.serialization import save_numpy


def partition_non_iid(X, y, roles, proc_dir):
    """
    Partition a global dataset into role-specific slices.

    Strategy:
        - randomly assign each client a disjoint subset of nodes
        - slice X/y accordingly

    Args:
        X: numpy array [N, seq_len, num_nodes]
        y: numpy array [N, num_nodes]
        roles: list of client names
        proc_dir: output directory where per-client folders will be written

    Returns:
        node_assignments: dict mapping role -> list of assigned node indices
    """
    num_nodes = X.shape[-1]
    num_roles = len(roles)

    # Random equal partition of node indices
    indices = np.arange(num_nodes)
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_roles)

    node_assignments = {}

    for role, nodes in zip(roles, splits):
        role_dir = os.path.join(proc_dir, role)
        os.makedirs(role_dir, exist_ok=True)

        # Slice X/y for assigned nodes
        X_slice = X[:, :, nodes]
        y_slice = y[:, nodes]

        save_numpy(os.path.join(role_dir, "X.npy"), X_slice)
        save_numpy(os.path.join(role_dir, "y.npy"), y_slice)

        node_assignments[role] = nodes.tolist()

    return node_assignments
