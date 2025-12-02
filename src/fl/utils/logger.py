"""
Lightweight logging helpers for federated learning runs.

Provides a simple JSONL logger and a timer utility for measuring latency
and durations throughout client and server execution.
"""

import os
import json
import time
from typing import Any, Dict

LOG_DIR = os.path.join("outputs", "logs")
os.makedirs(LOG_DIR, exist_ok=True)


class Timer:
    """Small helper for measuring elapsed wall-clock seconds."""

    def __init__(self):
        """Prepare timer with no active start."""
        self._start = None

    def start(self):
        """Start timing."""
        self._start = time.time()

    def stop(self):
        """Stop timing and return elapsed seconds."""
        if self._start is None:
            return 0.0
        elapsed = time.time() - self._start
        self._start = None
        return elapsed


def log_event(filename, event):
    """
    Append an event as a JSON line into outputs/logs/<filename>.

    Each event is enriched with a UNIX timestamp if missing.
    """
    path = os.path.join(LOG_DIR, filename)
    event = dict(event)
    event.setdefault("ts", time.time())
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")
