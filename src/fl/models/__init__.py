"""
Model definitions used in the federated learning experiments.

Currently exposes a lightweight GRU-based predictor for spatio-temporal
traffic flow forecasting.
"""

from .gru_model import SimpleGRU

__all__ = ["SimpleGRU"]
