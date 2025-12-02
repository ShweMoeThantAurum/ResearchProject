"""
Local differential privacy utilities for client updates.

This module wraps the Gaussian noise mechanism from src.fl.utils.privacy
and applies it conditionally based on environment variables.
"""

import os
from src.fl.utils.privacy import dp_add_noise


def dp_is_enabled():
    """
    Return True if differential privacy is enabled for this client.

    Reads DP_ENABLED from environment variables.
    """
    flag = os.environ.get("DP_ENABLED", "false").strip().lower()
    return flag in ["1", "true", "yes", "on"]


def get_dp_sigma():
    """
    Return the standard deviation of Gaussian noise for DP.

    Reads DP_SIGMA from environment variables, defaults to 0.01.
    """
    return float(os.environ.get("DP_SIGMA", "0.01"))


def maybe_add_dp_noise(state_dict):
    """
    Add Gaussian noise to model parameters if DP is enabled.

    Returns the possibly modified state_dict.
    """
    if not dp_is_enabled():
        return state_dict

    sigma = get_dp_sigma()
    noisy = dp_add_noise(state_dict, sigma)
    return noisy
