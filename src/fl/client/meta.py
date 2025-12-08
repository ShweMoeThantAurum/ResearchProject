"""
Metadata construction for client updates.

Each round, clients upload a JSON dictionary containing:
 - loss and sample counts
 - bandwidth and payload size
 - compute + communication energy
Used by AEFL for energy-aware selection.
"""


def compute_bandwidth_mbps(payload_bytes, upload_latency_sec):
    """Compute Mbps, returning 0.0 if latency is invalid."""
    if upload_latency_sec <= 0:
        return 0.0
    return (payload_bytes * 8 / 1e6) / upload_latency_sec


def build_round_metadata(
    role,
    round_id,
    energy_record,
    train_loss,
    train_samples,
    update_bytes,
    upload_latency_sec,
):
    """Build metadata dictionary uploaded to the server."""
    bw = compute_bandwidth_mbps(update_bytes, upload_latency_sec)

    return {
        "role": role,
        "round": round_id,
        "train_loss": float(train_loss),
        "train_samples": int(train_samples),
        # energy
        "compute_j_time": float(energy_record.get("compute_j_time", 0.0)),
        "compute_j_flops": float(energy_record.get("compute_j_flops", 0.0)),
        "comm_j_total": float(energy_record.get("comm_j_total", 0.0)),
        "total_energy_j": float(energy_record.get("total_energy_j", 0.0)),
        # communication meta
        "bandwidth_mbps": bw,
    }
