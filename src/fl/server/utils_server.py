"""
Utility helpers for server setup and file management.
"""

import os
import torch

from src.fl.models import SimpleGRU
from src.fl.config import settings


def ensure_dirs():
    """
    Ensure experiment directories exist.
    """
    dirs = [
        "experiments/updates",
        "experiments/metadata",
        "outputs/summaries",
    ]
    for d in dirs:
        if not os.path.exists(d):
            os.makedirs(d)


def init_global_model(hidden):
    """
    Initialize new GRU model and return state dict.
    """
    model = SimpleGRU(
        input_dim=1,
        hidden_size=hidden,
        num_layers=1
    )
    return model.state_dict()
