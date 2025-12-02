"""
Local differential privacy utilities for client updates.
Adds Gaussian noise to model parameters when enabled.
"""

import os
import torch


def _dp_enabled():
    """Return True if DP is enabled via environment variable."""
    return os.environ.get("DP_ENABLED", "false").strip().lower() == "true"


def _dp_sigma():
    """Return Gaussian noise standard deviation for DP."""
    return float(os.environ.get("DP_SIGMA", "0.01"))


def apply_dp_if_enabled(state_dict, role, round_id):
    """Optionally add Gaussian noise to model parameters."""
    if not _dp_enabled():
        return state_dict

    sigma = _dp_sigma()
    noisy = {}

    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            noise = torch.randn_like(v) * sigma
            noisy[k] = v + noise
        else:
            noisy[k] = v

    print(f"[{role}] Applied DP noise (sigma={sigma}) at round {round_id}")
    return noisy
