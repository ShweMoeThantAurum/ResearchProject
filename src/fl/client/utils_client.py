"""
Client-side helper utilities.

Provides:
- environment-derived constants
- simple bandwidth estimation
- optional local temp-file cleanup.
"""

import os
import glob
import os.path as osp

from src.fl.utils.logger import log_event


def compute_bandwidth_mbps(payload_bytes, upload_latency_sec):
    """Estimate uplink bandwidth in Mb/s from bytes and latency."""
    if upload_latency_sec <= 0.0:
        return 0.0
    mbits = payload_bytes * 8.0 / 1e6
    return mbits / upload_latency_sec


def cleanup_local_tmp(role):
    """Remove stale temporary model files for this role if any exist."""
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

    log_event(f"[{role}] cleanup removed_files={removed}")
