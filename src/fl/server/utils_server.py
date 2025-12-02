"""
Utility functions for server initialisation and configuration.

Handles:
- Determining processed dataset directory
- Inferring number of nodes
- Initialising global model
- Loading rounds and configs
"""

import os
import numpy as np
import torch

from src.fl.models import SimpleGRU
from src.fl.utils.serialization import save_torch, load_torch


def get_proc_dir(dataset):
    """Return path to processed data directory."""
    return os.path.join("datasets", "processed", dataset)


def infer_num_nodes(proc_dir):
    """Infer number of nodes from X_train.npy shape."""
    X = np.load(os.path.join(proc_dir, "X_train.npy"))
    return X.shape[-1]


def get_hidden_size():
    """Read HIDDEN_SIZE from environment."""
    return int(os.environ.get("HIDDEN_SIZE", 64))


def get_fl_rounds():
    """Read FL_ROUNDS from env."""
    return int(os.environ.get("FL_ROUNDS", 20))


def init_global_model(num_nodes, hidden):
    """Create a new global GRU model and return state dict."""
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden).cpu()
    return model.state_dict()


def store_global_model(state, round_id, dataset, mode):
    """Save global model to S3 and local outputs."""
    path = os.path.join("outputs", "models", dataset, mode.lower(), "global_round_%d.pth" % round_id)
    save_torch(path, state)
