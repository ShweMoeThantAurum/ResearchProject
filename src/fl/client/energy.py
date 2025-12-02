"""
Energy estimation utilities for client rounds.
Combines compute, FLOPs-based and communication energy.
"""

from ..utils.logger import log_event
from .utils_client import get_device_power_watts, get_net_j_per_mb


def estimate_round_energy(role,
                          round_id,
                          train_time_sec,
                          approx_flops,
                          download_bytes,
                          upload_bytes):
    """Estimate compute and communication energy for a round."""
    device_power = get_device_power_watts()
    net_j_per_mb = get_net_j_per_mb()

    # Wall-clock compute energy
    compute_time_j = device_power * train_time_sec

    # Very rough FLOPs energy model (constant per FLOP)
    j_per_flop = 1e-12
    compute_flops_j = approx_flops * j_per_flop

    total_bytes = download_bytes + upload_bytes
    mb = total_bytes / (1024.0 * 1024.0)
    comm_j = net_j_per_mb * mb

    total_j = compute_time_j + compute_flops_j + comm_j

    record = {
        "type": "client_energy",
        "role": role,
        "round": round_id,
        "train_time_sec": train_time_sec,
        "approx_flops": approx_flops,
        "download_bytes": download_bytes,
        "upload_bytes": upload_bytes,
        "compute_time_j": compute_time_j,
        "compute_flops_j": compute_flops_j,
        "comm_j": comm_j,
        "total_j": total_j,
    }

    log_event(record)

    print(
        f"[{role}] Energy r={round_id}: "
        f"compute_time={compute_time_j:.2f} J, "
        f"compute_flops={compute_flops_j:.6f} J, "
        f"comm_total={comm_j:.4f} J, "
        f"total={total_j:.4f} J"
    )

    return record
