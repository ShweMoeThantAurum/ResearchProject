"""
Utility helpers used by the federated learning server.

This module keeps purely local logic:
  - inferring dataset paths
  - inferring number of nodes
  - constructing the GRU model skeleton
"""

import os
import numpy as np
from src.fl.models.gru_model import SimpleGRU


def get_dataset_name():
    """
    Return the active dataset name as a lowercase string.

    Uses the DATASET environment variable, defaulting to "sz".
    """
    return os.environ.get("DATASET", "sz").strip().lower()


def get_proc_dir(dataset_name):
    """
    Build the path to the preprocessed dataset directory for a dataset.

    The convention is:
        datasets/processed/<dataset_name>/prepared
    """
    return os.path.join("datasets", "processed", dataset_name, "prepared")


def infer_num_nodes(proc_dir):
    """
    Infer the number of nodes from the shape of X_train.npy.

    Returns the last dimension of X_train as an integer.
    """
    x_path = os.path.join(proc_dir, "X_train.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(
            "Cannot infer number of nodes. Missing X_train at: " + x_path
        )

    X = np.load(x_path)
    return int(X.shape[-1])


def build_initial_model(num_nodes, hidden_size):
    """
    Construct a SimpleGRU model and return its state dictionary.

    The server never trains this model locally. It only uses the
    state dictionary as the initial global model for round 1.
    """
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size)
    return model.state_dict()
