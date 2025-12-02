"""
Central lightweight logging utilities for FL experiments.
Supports timestamped logs for server and clients.
"""

import os
import time
from datetime import datetime


LOG_DIR = "outputs/logs"


def ensure_log_dir():
    """Create log directory if missing."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


def timestamp():
    """Return current UTC timestamp."""
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")


def log_event(message, logfile="events.log"):
    """Write a timestamped event into logs/."""
    ensure_log_dir()
    path = os.path.join(LOG_DIR, logfile)
    with open(path, "a") as f:
        f.write(f"[{timestamp()}] {message}\n")


class Timer:
    """Simple context timer for profiling code sections."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.duration = self.end - self.start
