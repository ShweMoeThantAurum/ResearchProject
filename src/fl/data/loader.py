"""
Convenience loaders for processed datasets.

Files expected under:
    datasets/processed/<dataset>/

Core files:
    - X_train.npy
    - y_train.npy
    - X_valid.npy
    - y_valid.npy
    - X_test.npy
    - y_test.npy

Per-role files (created by preprocessing):
    - X_<role>.npy
    - y_<role>.npy
"""

import os
import numpy as np
from src.fl.data.partition import DEFAULT_ROLES


def get_processed_dir(dataset, root="datasets/processed"):
    """
    Return the processed directory for a dataset, e.g.

        datasets/processed/sz
        datasets/processed/los
        datasets/processed/pems08
    """
    return os.path.join(root, dataset)


def _load_numpy(path):
    if not os.path.exists(path):
        raise FileNotFoundError("Missing %s" % path)
    return np.load(path)


def load_splits(dataset, root="datasets/processed"):
    """
    Load train/valid/test splits for a dataset.

    Returns dict:
        {
            "X_train": ...,
            "y_train": ...,
            "X_valid": ...,
            "y_valid": ...,
            "X_test": ...,
            "y_test": ...
        }
    """
    proc_dir = get_processed_dir(dataset, root=root)

    data = {}
    for split in ["train", "valid", "test"]:
        X_path = os.path.join(proc_dir, "X_%s.npy" % split)
        y_path = os.path.join(proc_dir, "y_%s.npy" % split)
        data["X_%s" % split] = _load_numpy(X_path)
        data["y_%s" % split] = _load_numpy(y_path)

    return data


def load_role_data(dataset, role, root="datasets/processed"):
    """
    Load per-role training data.

    Returns:
        X_role, y_role
    """
    proc_dir = get_processed_dir(dataset, root=root)

    X_path = os.path.join(proc_dir, "X_%s.npy" % role)
    y_path = os.path.join(proc_dir, "y_%s.npy" % role)

    X = _load_numpy(X_path)
    y = _load_numpy(y_path)

    return X, y


def list_available_roles(dataset, root="datasets/processed"):
    """
    Check which roles have X_<role>.npy present.
    """
    proc_dir = get_processed_dir(dataset, root=root)
    roles = []
    for role in DEFAULT_ROLES:
        X_path = os.path.join(proc_dir, "X_%s.npy" % role)
        if os.path.exists(X_path):
            roles.append(role)
    return roles
