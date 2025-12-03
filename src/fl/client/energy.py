"""
Energy estimation utilities for FL clients.
Models compute cost (GRU forward/backward) and communication cost.
"""

import torch
import time


def estimate_compute_energy(model, batch_count, device_power_watts):
    """
    Rough compute energy model:
    energy = power (W) * time (s)
    """
    start = time.time()

    # small dummy forward pass to approximate cost per batch
    dummy = torch.randn(1, model.seq_len, model.num_nodes, device=model.fc.weight.device)
    _ = model(dummy)

    elapsed = time.time() - start
    total_time = elapsed * batch_count
    return device_power_watts * total_time


def estimate_comm_energy(bytes_up, net_j_per_mb):
    """
    Simple linear model for wireless transmission cost.
    """
    mb = bytes_up / (1024 * 1024)
    return mb * net_j_per_mb


def estimate_energy(model,
                    batch_count,
                    bytes_up,
                    device_power_watts,
                    net_j_per_mb):
    """
    Main entry used by client_main.
    Computes total energy = compute + communication.
    """
    compute_j = estimate_compute_energy(model, batch_count, device_power_watts)
    comm_j = estimate_comm_energy(bytes_up, net_j_per_mb)
    total_j = compute_j + comm_j

    return {
        "compute_j": compute_j,
        "comm_j": comm_j,
        "total_j": total_j,
    }
