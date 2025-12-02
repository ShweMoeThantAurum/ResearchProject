"""
Client package for federated learning experiments.

Exposes the main entrypoint that each Docker client container runs.
"""

from .client_main import run_client


__all__ = ["run_client"]
