"""
Lightweight logging utilities for FL training and evaluation.
Handles simple timestamped events and timing helpers.
"""

import os
import time
from datetime import datetime


LOG_DIR = "outputs/logs"


def _ensure_log_dir():
    """Create log directory if it does not exist."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


class Timer:
    """Simple context timer for measuring durations."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.time()
        self.duration = self.end - self.start


def log_event(message, logfile="events.log"):
    """Append a timestamped message into the central log file."""
    _ensure_log_dir()
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    path = os.path.join(LOG_DIR, logfile)
    with open(path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")
