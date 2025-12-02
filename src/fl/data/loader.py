"""
Dataset loading utilities for federated learning experiments.

This module handles:
    - Loading global train/validation/test arrays from disk
    - Inspecting dataset metadata (number of nodes, shapes)
    - Loading per-client partitions created by the partition module
"""

import os
import numpy as np


def get_processed_dir(dataset_name):
    """
    Return the processed dataset directory for the given dataset name.

    Example:
        dataset_name = "sz"
        -> "datasets/processed/sz"
    """
    return os.path.join("datasets", "processed", dataset_name)


def load_global_arrays(proc_dir):
    """
    Load global train/validation/test arrays from a processed directory.

    Expects files:
        X_train.npy, y_train.npy
        X_valid.npy, y_valid.npy
        X_test.npy,  y_test.npy

    Returns a dictionary:
        {
            "X_train": np.ndarray,
            "y_train": np.ndarray,
            "X_valid": np.ndarray,
            "y_valid": np.ndarray,
            "X_test":  np.ndarray,
            "y_test":  np.ndarray,
        }
    """
    splits = ["train", "valid", "test"]
    data = {}

    for split in splits:
        x_path = os.path.join(proc_dir, "X_%s.npy" % split)
        y_path = os.path.join(proc_dir, "y_%s.npy" % split)

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            raise FileNotFoundError("Missing X_%s or y_%s in %s" %
                                    (split, split, proc_dir))

        data["X_%s" % split] = np.load(x_path)
        data["y_%s" % split] = np.load(y_path)

    return data


def get_num_nodes(proc_dir):
    """
    Infer the number of graph nodes from X_train.

    Assumes X_train has shape:
        [num_samples, seq_len, num_nodes]
    """
    x_path = os.path.join(proc_dir, "X_train.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError("Missing X_train.npy in %s" % proc_dir)

    x = np.load(x_path)
    return int(x.shape[-1])


def client_partition_dir(proc_dir):
    """
    Return the clients/ directory inside a processed dataset directory.
    """
    return os.path.join(proc_dir, "clients")


def get_client_file_paths(proc_dir, role):
    """
    Return file paths for a given client role.

    Example:
        role = "vehicle"
        -> "clients/client_vehicle_X.npy"
           "clients/client_vehicle_y.npy"
    """
    root = client_partition_dir(proc_dir)
    x_path = os.path.join(root, "client_%s_X.npy" % role)
    y_path = os.path.join(root, "client_%s_y.npy" % role)
    return x_path, y_path


def load_client_partition(proc_dir, role):
    """
    Load a client's local partition arrays (X, y) for a given role.

    Returns:
        X_local: np.ndarray [samples, seq_len, num_nodes_local]
        y_local: np.ndarray [samples, num_nodes_local]
    """
    x_path, y_path = get_client_file_paths(proc_dir, role)

    if not os.path.exists(x_path):
        raise FileNotFoundError("Missing client X file: %s" % x_path)
    if not os.path.exists(y_path):
        raise FileNotFoundError("Missing client y file: %s" % y_path)

    X = np.load(x_path)
    y = np.load(y_path)

    return X, y
