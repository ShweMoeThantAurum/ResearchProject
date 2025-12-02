"""
Module entrypoint for server execution.

Allows the server to be started using:
    python -m src.fl.server
"""

from .server_main import main

if __name__ == "__main__":
    main()
