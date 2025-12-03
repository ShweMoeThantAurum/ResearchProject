"""
Client-side helpers for configuration, metadata, and housekeeping.
Provides role/dataset access and metadata builders.
"""

import os
import glob
import os.path as osp

from ..utils.logger import log_event


def get_role():
    """Return client role name."""
    return os.environ.get("CLIENT_ROLE", "roadside").strip().lower()


def get_dataset():
    """Return dataset name."""
    return os.environ.get("DATASET", "sz").strip().lower()


def compute_bandwidth_mbps(payload_bytes, upload_latency_sec):
    """Compute simple uplink bandwidth estimate."""
    if upload_latency_sec <= 0.0:
        return 0.0
    mbits = payload_bytes * 8.0 / 1e6
    return mbits / upload_latency_sec


def build_round_metadata(role,
                         round_id,
                         train_loss,
                         train_samples,
                         compute_time_j,
                         compute_flops_j,
                         comm_j,
                         download_bytes,
                         upload_bytes,
                         upload_latency_sec):
    """Build metadata dictionary for server-side AEFL selection."""
    bw_mbps = compute_bandwidth_mbps(upload_bytes, upload_latency_sec)

    return {
        "role": role,
        "round": round_id,
        "train_loss": float(train_loss),
        "train_samples": int(train_samples),
        "compute_time_j": float(compute_time_j),
        "compute_flops_j": float(compute_flops_j),
        "comm_j": float(comm_j),
        "total_energy_j": float(compute_time_j + compute_flops_j + comm_j),
        "download_mb": float(download_bytes / (1024.0 * 1024.0)),
        "upload_mb": float(upload_bytes / (1024.0 * 1024.0)),
        "bandwidth_mbps": float(bw_mbps),
    }


def cleanup_local_tmp(role):
    """Remove stale temporary model files for this role."""
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
