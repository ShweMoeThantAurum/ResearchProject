"""
Shared server utilities: roles, config helpers, paths.
"""

import os

ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def _get_env(name, default):
    """Reads environment variable with fallback."""
    v = os.environ.get(name, default)
    return v


def get_mode():
    """Returns the FL mode (AEFL, FedAvg, FedProx, LocalOnly)."""
    return _get_env("FL_MODE", "AEFL").lower()


def is_aefl(mode):
    return mode.lower() == "aefl"


def is_fedavg(mode):
    return mode.lower() == "fedavg"


def is_fedprox(mode):
    return mode.lower() == "fedprox"


def is_localonly(mode):
    return mode.lower() == "localonly"


def get_fl_rounds():
    """Returns number of FL rounds."""
    return int(_get_env("FL_ROUNDS", "20"))


def get_hidden_size():
    """Returns GRU hidden size."""
    return int(_get_env("HIDDEN_SIZE", "64"))


def get_batch_size():
    """Returns batch size."""
    return int(_get_env("BATCH_SIZE", "64"))


def get_proc_dir():
    """Global directory used to store round updates."""
    return "datasets/processed"
