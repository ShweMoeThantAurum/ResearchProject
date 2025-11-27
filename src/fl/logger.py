"""Lightweight logging utilities for federated learning experiments."""

import os
import json
import time
from typing import Any, Dict

LOG_DIR = "run_logs"
os.makedirs(LOG_DIR, exist_ok=True)


class Timer:
    """Simple wall-clock timer used to measure elapsed time in seconds."""

    def __init__(self):
        """Initialise the timer with no active start time."""
        self._start = None

    def start(self):
        """Start the timer."""
        self._start = time.time()

    def stop(self):
        """
        Stop the timer and return the elapsed time in seconds.

        If the timer was never started, returns 0.0.
        """
        if self._start is None:
            return 0.0
        elapsed = time.time() - self._start
        self._start = None
        return elapsed


def log_event(filename: str, event: Dict[str, Any]) -> None:
    """
    Append a single JSON event to a log file in JSON Lines (JSONL) format.

    The event is stored under run_logs/<filename>, with an added timestamp
    field 'ts' in seconds since the Unix epoch.
    """
    path = os.path.join(LOG_DIR, filename)
    event = dict(event)
    event.setdefault("ts", time.time())
    with open(path, "a") as f:
        f.write(json.dumps(event) + "\n")
