"""
Partition processed datasets into per-client splits.
"""

import numpy as np
from src.fl.data.loader import load_prepared_data


CLIENT_ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def split_equally(x, y, n):
    """Split arrays equally into n segments."""
    size = len(x)
    seg = size // n

    xs = []
    ys = []

    for i in range(n):
        start = i * seg
        end = (i + 1) * seg
        xs.append(x[start:end])
        ys.append(y[start:end])

    return xs, ys


def build_client_partitions():
    """Return a mapping from role to its local train/test partitions."""
    train_x, train_y, test_x, test_y = load_prepared_data()

    # Equal partitioning for all 5 roles
    xs, ys = split_equally(train_x, train_y, len(CLIENT_ROLES))

    client_data = {}
    for idx, role in enumerate(CLIENT_ROLES):
        client_data[role] = {
            "train_x": xs[idx],
            "train_y": ys[idx],
            "test_x": test_x,
            "test_y": test_y,
        }

    return client_data
 