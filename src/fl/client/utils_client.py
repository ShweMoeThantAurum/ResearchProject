"""
Client-side utility functions.

Provides:
    - loading local dataset partitions
    - simple helpers used by client_main
"""

import os
import numpy as np


def load_client_data(proc_dir, role):
    """
    Load local client data arrays for a given IoT device role.

    Files:
        client_<role>_X.npy
        client_<role>_y.npy

    Example:
        role = "vehicle"
        -> clients/client_vehicle_X.npy
           clients/client_vehicle_y.npy
    """
    x_path = os.path.join(proc_dir, "clients", "client_%s_X.npy" % role)
    y_path = os.path.join(proc_dir, "clients", "client_%s_y.npy" % role)

    if not os.path.exists(x_path):
        raise FileNotFoundError("Client X file not found: %s" % x_path)
    if not os.path.exists(y_path):
        raise FileNotFoundError("Client y file not found: %s" % y_path)

    X = np.load(x_path)
    y = np.load(y_path)

    return X, y
