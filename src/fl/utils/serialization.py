"""
Serialization helpers for model states, tensors, and experiment outputs.

Provides wrappers around PyTorch and NumPy save/load utilities while
ensuring directories are created automatically.
"""

import os
import json
import numpy as np
import torch


def ensure_dir(path):
    """Create directory if it does not already exist."""
    os.makedirs(os.path.dirname(path), exist_ok=True)


def save_json(path, data):
    """Save a Python dictionary as a JSON file."""
    ensure_dir(path)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """Load and return JSON content from a file."""
    with open(path, "r") as f:
        return json.load(f)


def save_numpy(path, array):
    """Save a NumPy array to disk."""
    ensure_dir(path)
    np.save(path, array)


def load_numpy(path):
    """Load a NumPy array from disk."""
    return np.load(path)


def save_torch(path, state_dict):
    """Save a PyTorch state dict to disk."""
    ensure_dir(path)
    torch.save(state_dict, path)


def load_torch(path):
    """Load a PyTorch state dict."""
    return torch.load(path, map_location="cpu")
