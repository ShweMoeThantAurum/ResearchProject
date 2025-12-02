"""
Misc client-side utilities.
"""

import numpy as np
import sys
import os


def print_flush(msg):
    """Print with immediate stdout flush (good for docker logs)."""
    print(msg)
    sys.stdout.flush()


def load_client_dataset(proc_dir, role):
    """
    Each client loads:
        X_<role>.npy
        y_<role>.npy
    """
    X_path = os.path.join(proc_dir, "X_%s.npy" % role)
    y_path = os.path.join(proc_dir, "y_%s.npy" % role)

    X = np.load(X_path)
    y = np.load(y_path)

    return X, y
