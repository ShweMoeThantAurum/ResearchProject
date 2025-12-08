"""
Client-side Differential Privacy (DP) utilities.

Adds Gaussian noise to model parameters when DP is enabled via
environment variables.
"""

import os
from src.utils.privacy import dp_add_noise


def maybe_add_dp_noise(state_dict):
    """
    Apply local Gaussian DP noise if DP is enabled.
    """
    if os.environ.get("DP_ENABLED", "false").lower() != "true":
        return state_dict

    sigma = float(os.environ.get("DP_SIGMA", "0.01"))
    return dp_add_noise(state_dict, sigma)
