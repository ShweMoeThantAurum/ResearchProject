"""
Client package initialiser for federated learning.

Exposes the main client entrypoint used by Docker containers
during cloud-based FL execution.
"""

from .main import main

__all__ = ["main"]
