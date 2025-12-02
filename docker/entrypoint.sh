#!/usr/bin/env bash
# Entrypoint for client containers. Runs FL client process.

set -e

echo "[entrypoint] Starting client for role=$CLIENT_ROLE mode=$FL_MODE dataset=$DATASET"

exec python -m src.fl.client
