"""
Server package initializer for the federated learning pipeline.

Exposes the main server entrypoint used for cloud-based FL orchestration.
"""

from .main import main

__all__ = ["main"]
