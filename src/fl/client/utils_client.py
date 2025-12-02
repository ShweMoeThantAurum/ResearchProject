"""
Client-side helpers for configuration, metadata, and housekeeping.
Reads environment variables and builds per-round metadata.
"""

import os
import glob
import os.path as osp

from ..utils.logger import log_event


def get_role():
    """Return client role name."""
    return os.environ.get("CLIENT_ROLE", "roadside").strip()


def get_dataset():
    """Return dataset name."""
    return os.environ.get("DATASET", "sz").strip().lower()


def get_fl_mode():
    """Return FL mode (AEFL, FedAvg, FedProx, LocalOnly)."""
    return os.environ.get("FL_MODE", "AEFL").strip()


def get_fl_rounds():
    """Return number of federated rounds."""
    return int(os.environ.get("FL_ROUNDS", "20"))


def get_batch_size():
    """Return local mini-batch size."""
    return int(os.environ.get("BATCH_SIZE", "64"))


def get_local_epochs():
    """Return number of local epochs per round."""
    return int(os.environ.get("LOCAL_EPOCHS", "1"))


def get_lr():
    """Return client learning rate."""
    return float(os.environ.get("LR", "0.001"))


def get_hidden_size():
    """Return GRU hidden size."""
    return int(os.environ.get("HIDDEN_SIZE", "64"))


def get_device_power_watts():
    """Return average device power consumption."""
    return float(os.environ.get("DEVICE_POWER_WATTS", "3.5"))


def get_net_j_per_mb():
    """Return energy cost per MB of communication."""
    return float(os.environ.get("NET_J_PER_MB", "0.6"))


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
