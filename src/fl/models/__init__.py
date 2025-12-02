"""
Model definitions for the federated learning framework.

Currently exposes:
    - SimpleGRU : lightweight GRU-based traffic predictor
"""

from .gru_model import SimpleGRU

__all__ = ["SimpleGRU"]
