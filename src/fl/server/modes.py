"""Helpers determining which FL mode (AEFL/FedAvg/FedProx/LocalOnly) is active."""

import os


def get_mode():
    """
    Return the active FL mode in lowercase.
    Defaults to AEFL.
    """
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    return mode


def is_aefl(mode: str):
    return mode.lower() == "aefl"


def is_fedavg(mode: str):
    return mode.lower() == "fedavg"


def is_fedprox(mode: str):
    return mode.lower() == "fedprox"


def is_localonly(mode: str):
    return mode.lower() == "localonly"
