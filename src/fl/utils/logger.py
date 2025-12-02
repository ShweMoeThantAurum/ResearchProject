"""
Lightweight logging utilities for federated learning runs.
"""

import os
import time

# All plain-text logs go under this directory.
LOG_DIR = "outputs/logs"


def _ensure_log_dir():
    """Create the log directory if it does not exist."""
    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)


def log_event(message, role=None):
    """Append a log line to a file and print it to the console."""
    _ensure_log_dir()
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    # Optional role prefix such as [roadside], [server], etc.
    prefix = "[{}]".format(role) if role else ""
    line = "{} {} {}\n".format(ts, prefix, message)

    path = os.path.join(LOG_DIR, "events.log")
    with open(path, "a") as f:
        f.write(line)

    # Mirror logs to stdout for real-time observation.
    print(line.strip())


class Timer:
    """Measure elapsed wall-clock time."""

    def __init__(self):
        """Start the timer."""
        self.start = time.time()

    def elapsed(self):
        """Return elapsed time in seconds."""
        return time.time() - self.start
