"""
Utility functions for the client.
"""

import os
import json
import numpy as np


def load_client_data(dataset, role):
    """
    Load preprocessed dataset partition for this client.
    """
    path = "datasets/processed/{}/{}/data.npz".format(dataset, role)

    if not os.path.exists(path):
        raise FileNotFoundError("Client data missing: {}".format(path))

    data = np.load(path, allow_pickle=True)
    return data["x"], data["y"]


def decode_state_dict(encoded):
    """
    Decode a JSON state dict into tensors.
    """
    state = {}
    for k, arr in encoded.items():
        state[k] = torch.tensor(arr)
    return state
