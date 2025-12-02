"""
Client package initializer.

Exposes the main federated learning client entrypoint that is used
inside Docker containers.
"""

from .client_main import run_client


__all__ = ["run_client"]
