"""
Local differential privacy (DP) utilities for client updates.

Adds light Gaussian noise to model parameters when DP is enabled
via environment variables.
"""

import os
from typing import Dict
import torch
from src.utils.privacy import dp_add_noise


def maybe_add_dp_noise(state_dict):
    """Add small Gaussian noise to model updates if DP is enabled."""
    enabled = os.environ.get("DP_ENABLED", "false").lower() == "true"
    if not enabled:
        return state_dict

    sigma = float(os.environ.get("DP_SIGMA", "0.01"))
    return dp_add_noise(state_dict, sigma)
