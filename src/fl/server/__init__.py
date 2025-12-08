"""
Server package initializer for the federated learning pipeline.

This package exposes the main server entry point used to orchestrate
cloud-based federated learning experiments.
"""

from .main import main

__all__ = ["main"]
