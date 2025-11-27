"""Client-side cleanup utilities for removing stale temporary files."""

import glob
import os
from src.fl.logger import log_event


def cleanup_local_tmp(role: str):
    """
    Remove temporary FL files from previous runs for the given client role.

    Deletes local /tmp/global_<role>_*.pt and /tmp/update_<role>_*.pt files
    to ensure that each new run starts clean.
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
                pass

    log_event("client_cleanup.log", {"role": role, "removed_files": removed})

    if removed > 0:
        print(f"[{role}] Cleaned up {removed} local tmp files.")
    else:
        print(f"[{role}] No local tmp files to clean.")
