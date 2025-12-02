"""
I/O helpers for reading updates and writing global models.
No real S3 usage; local file emulation only.
"""

import os
import json
import base64
import torch

from src.fl.utils.serialization import decode_state_dict, encode_state_dict


def save_global_state(state, round_id):
    """
    Save a global model state for the next round.
    """
    path = "experiments/global_model_r{}.bin".format(round_id)
    encoded = encode_state_dict(state)

    with open(path, "w") as f:
        json.dump(encoded, f)


def load_update(role, round_id):
    """
    Load a client update if it exists.
    """
    path = "experiments/updates/round_{}/{}.json".format(round_id, role)

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        return decode_state_dict(json.load(f))


def load_round_metadata(round_id):
    """
    Load client metadata for AEFL client selection.
    """
    directory = "experiments/metadata"
    if not os.path.exists(directory):
        return {}

    result = {}
    for fname in os.listdir(directory):
        if fname.endswith("_r{}.json".format(round_id)):
            role = fname.split("_r")[0]
            with open(os.path.join(directory, fname), "r") as f:
                result[role] = json.load(f)

    return result
