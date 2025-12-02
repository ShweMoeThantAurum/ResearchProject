#!/bin/bash

echo "[ENTRYPOINT] Starting container for role: $CLIENT_ROLE"

# Fail if role is missing
if [ -z "$CLIENT_ROLE" ] && [ "$1" = "client" ]; then
    echo "[ERROR] CLIENT_ROLE must be set for client containers."
    exit 1
fi

# Server mode
if [ "$1" = "server" ]; then
    echo "[ENTRYPOINT] Launching FL server..."
    exec python -m src.fl.server_main
fi

# Client mode
if [ "$1" = "client" ]; then
    echo "[ENTRYPOINT] Launching FL client ($CLIENT_ROLE)..."
    exec python -m src.fl.client_main
fi

echo "[ENTRYPOINT] Unknown mode. Expected 'client' or 'server'."
exit 1
