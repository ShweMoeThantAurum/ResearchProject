"""Client-side metadata builder for reporting energy + bandwidth usage."""

def compute_bandwidth_mbps(payload_bytes: int, upload_latency_sec: float):
    """
    Compute uplink bandwidth in megabits per second from payload
    size and upload time.
    """
    if upload_latency_sec <= 0.0:
        return 0.0
    mbits = (payload_bytes * 8.0) / 1e6
    return mbits / upload_latency_sec


def build_round_metadata(role: str,
                         round_id: int,
                         energy_record: dict,
                         train_loss: float,
                         train_samples: int,
                         update_bytes: int,
                         upload_latency_sec: float):
    """
    Build the metadata dictionary uploaded by each client.

    Includes energy, communication, bandwidth, and training statistics.
    """
    bw_mbps = compute_bandwidth_mbps(update_bytes, upload_latency_sec)

    return {
        "role": role,
        "round": round_id,
        "train_loss": float(train_loss),
        "train_samples": int(train_samples),
        "compute_j": float(energy_record.get("compute_j", 0.0)),
        "comm_j": float(energy_record.get("comm_j", 0.0)),
        "total_energy_j": float(energy_record.get("total_j", 0.0)),
        "download_mb": float(energy_record.get("download_mb", 0.0)),
        "upload_mb": float(energy_record.get("upload_mb", 0.0)),
        "bandwidth_mbps": bw_mbps,
    }
