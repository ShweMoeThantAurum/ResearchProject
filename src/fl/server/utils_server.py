"""
Server utility functions (config, directories, model init).
"""

import os
import shutil


def clear_round_data(dataset, mode):
    """Delete old experiment data in outputs/summaries and S3."""
    exp_dir = os.path.join("outputs", "summaries", dataset, mode)
    if os.path.exists(exp_dir):
        shutil.rmtree(exp_dir)
    os.makedirs(exp_dir, exist_ok=True)


def get_mode():
    return os.environ.get("FL_MODE", "AEFL").lower()


def get_fl_rounds():
    return int(os.environ.get("FL_ROUNDS", 20))


def get_hidden_size():
    return int(os.environ.get("HIDDEN_SIZE", 64))


def get_processed_dir(dataset):
    return os.path.join("datasets", "processed", dataset)


def infer_num_nodes(proc_dir):
    """Infer number of nodes from training data shape."""
    X_train = os.path.join(proc_dir, "X_train.npy")
    import numpy as np
    x = np.load(X_train)
    return x.shape[-1]
