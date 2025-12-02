"""
Server package initializer.

Exposes the main federated learning server entrypoint.
"""

from .server_main import run_server


__all__ = ["run_server"]
