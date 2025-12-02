"""
Differential privacy noise addition for model updates.
"""

import torch
import numpy as np

from src.fl.config import settings


def apply_dp(update):
    """
    Add Gaussian noise to parameters.
    """
    sigma = settings.get_dp_sigma()

    new_update = {}
    for k, v in update.items():
        noise = torch.randn_like(v) * sigma
        new_update[k] = v + noise

    return new_update
