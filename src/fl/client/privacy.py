"""Optional lightweight differential privacy for client model updates."""

import os
import torch
from typing import Dict
from src.utils.privacy import dp_add_noise


def maybe_add_dp_noise(state_dict: Dict[str, torch.Tensor]):
    """
    Add very light Gaussian noise to model updates if DP is enabled.

    Controlled by env vars:
      DP_ENABLED=true/false
      DP_SIGMA=<float>
    """
    enabled = os.environ.get("DP_ENABLED", "false").lower() == "true"
    if not enabled:
        return state_dict

    sigma = float(os.environ.get("DP_SIGMA", "0.01"))
    return dp_add_noise(state_dict, sigma)
    