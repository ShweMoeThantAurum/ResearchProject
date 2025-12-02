"""
Client communication utilities for downloading global models
and uploading updates + metadata.
"""

import os
import json
import base64
import torch

from src.fl.utils.serialization import encode_state_dict, decode_state_dict


def _global_model_path(round_id):
    return "experiments/global_model_r{}.bin".format(round_id)


def load_global_state(round_id):
    """
    Load the global model state saved by the server.
    """
    path = _global_model_path(round_id)
    if not os.path.exists(path):
        return None

    with open(path, "rb") as f:
        data = f.read()

    return json.loads(data)


def upload_update(update, role, round_id):
    """
    Save client update for server to consume.
    """
    out_dir = "experiments/updates/round_{}/".format(round_id)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    encoded = encode_state_dict(update)

    with open(os.path.join(out_dir, "{}.json".format(role)), "w") as f:
        json.dump(encoded, f)


def upload_metadata(role, round_id, bandwidth_mbps, total_energy):
    """
    Save client metadata for AEFL selection.
    """
    out_dir = "experiments/metadata/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    data = {
        "bandwidth_mbps": bandwidth_mbps,
        "total_energy_j": total_energy
    }

    with open(os.path.join(out_dir, "{}_r{}.json".format(role, round_id)), "w") as f:
        json.dump(data, f)
