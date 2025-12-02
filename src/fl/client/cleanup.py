"""
Cleanup utilities for removing stale temporary files created by past runs.

Ensures each federated learning round begins with a clean local state.
"""

import glob
import os
from src.fl.logger import log_event


def cleanup_local_tmp(role):
    """Remove old cached global/update files for this client role."""
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
                pass

    log_event("client_cleanup.log", {"role": role, "removed_files": removed})

    if removed > 0:
        print(f"[{role}] Cleaned up {removed} local tmp files.")
    else:
        print(f"[{role}] No local tmp files to clean.")
