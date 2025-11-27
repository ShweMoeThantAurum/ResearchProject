"""Defines client-side behaviour depending on the chosen FL mode."""

import os

VALID_MODES = ["AEFL", "FedAvg", "FedProx", "LocalOnly"]


def get_client_mode():
    """
    Return the federated learning mode for this client.

    Reads FL_MODE from the environment and defaults to AEFL.
    """
    mode = os.environ.get("FL_MODE", "AEFL").strip()
    if mode not in VALID_MODES:
        print(f"[CLIENT] WARNING: invalid FL_MODE='{mode}', using AEFL.")
        return "AEFL"
    return mode


def client_allows_training(mode: str):
    """
    Determine if the client should perform local training.

    All modes currently allow training (LocalOnly training is still performed).
    """
    return True
