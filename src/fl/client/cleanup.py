"""
Temporary-file cleanup utilities for client nodes.

Removes leftover cached model files to prevent interference between
independent experiments.
"""

import glob
import os
from src.fl.logger import log_event


def cleanup_local_tmp(role):
    """
    Remove temporary global/update model files for this client role.

    Called once at startup before FL rounds begin.
    """
    patterns = [
        f"/tmp/global_{role}_round_*.pt",
        f"/tmp/update_{role}_round_*.pt",
    ]

    removed = 0
    for pattern in patterns:
        for path in glob.glob(pattern):
            try:
                os.remove(path)
                removed += 1
            except Exception:
                pass  # best-effort cleanup

    log_event("client_cleanup.log", {"role": role, "removed": removed})

    msg = (
        f"[{role}] Cleaned {removed} tmp files."
        if removed
        else f"[{role}] No tmp files to clean."
    )
    print(msg)
