"""
Utility helpers for saving and loading model state dictionaries.
Thin wrappers around PyTorch serialization.
"""

import os
import json
import torch


def save_state(path, state_dict):
    """Save a PyTorch state_dict to disk."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    torch.save(state_dict, path)


def load_state(path):
    """Load a PyTorch state_dict from disk."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"State dict not found: {path}")
    return torch.load(path, map_location="cpu")


def save_json(path, data):
    """Write JSON data to a file."""
    directory = os.path.dirname(path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """Read JSON data from a file."""
    with open(path, "r") as f:
        return json.load(f)
