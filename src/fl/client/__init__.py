"""
Client package initializer for the federated learning pipeline.

Exposes the main client entry point used by containerised client nodes
during cloud-based federated learning experiments.
"""

from .main import main

__all__ = ["main"]
