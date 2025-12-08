"""
Lightweight logging helpers for federated learning runs.

Provides:
- a simple JSONL logger that writes into run_logs/
- a small wall-clock timer used throughout client and server code
"""

import os
import json
import time

LOG_DIR = "run_logs"
os.makedirs(LOG_DIR, exist_ok=True)


class Timer:
    """Tiny helper for measuring elapsed wall-clock time in seconds."""

    def __init__(self):
        """Initialise the timer without a running start time."""
        self._start = None

    def start(self):
        """Start timing from the current moment."""
        self._start = time.time()

    def stop(self):
        """
        Stop timing and return elapsed seconds.

        If the timer was never started, returns 0.0.
        """
        if self._start is None:
            return 0.0
        elapsed = time.time() - self._start
        self._start = None
        return elapsed


def log_event(filename, event):
    """
    Append an event dictionary as a JSON line into run_logs/<filename>.

    A timestamp (ts) is automatically added if missing.
    """
    path = os.path.join(LOG_DIR, filename)
    event = dict(event)
    event.setdefault("ts", time.time())

    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")
