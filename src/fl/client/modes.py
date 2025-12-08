"""
Client-side helpers for determining active FL mode.

Supports three modes:
 - AEFL      (Adaptive Energy-aware Federated Learning)
 - FedAvg
 - FedProx
"""

import os

VALID_MODES = ["aefl", "fedavg", "fedprox"]


def get_client_mode():
    """
    Read FL_MODE from environment, return validated lowercase string.

    Defaults to AEFL if misconfigured.
    """
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    if mode not in VALID_MODES:
        print(f"[CLIENT] WARNING: invalid FL_MODE='{mode}'. Using AEFL.")
        return "aefl"
    return mode


def client_allows_training(mode):
    """
    Return True if this client should run local training.

    Currently True for all modes (ablation support in future).
    """
    return True
