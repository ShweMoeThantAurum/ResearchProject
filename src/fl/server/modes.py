"""
Helpers for determining which FL mode the server is running in.

Provides simple boolean checks for:
 - AEFL   (Adaptive Energy-aware Federated Learning)
 - FedAvg
 - FedProx
"""

import os


def get_mode():
    """Return the active FL mode in lowercase (defaults to 'aefl')."""
    return os.environ.get("FL_MODE", "AEFL").strip().lower()


def is_aefl(mode):
    """Return True if the given mode corresponds to AEFL."""
    return mode.lower() == "aefl"


def is_fedavg(mode):
    """Return True if the given mode corresponds to FedAvg."""
    return mode.lower() == "fedavg"


def is_fedprox(mode):
    """Return True if the given mode corresponds to FedProx."""
    return mode.lower() == "fedprox"
