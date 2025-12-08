"""
Model definitions used in the federated learning experiments.

Currently exposes a lightweight GRU-based predictor for spatio-temporal
traffic flow forecasting across all FL modes.
"""

from .simple_gru import SimpleGRU

__all__ = ["SimpleGRU"]
