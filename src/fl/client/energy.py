"""
Energy model for federated client training and communication.

Energy components:
    - compute_time_j  (device wattage * time)
    - compute_flops_j (approx FLOP cost of GRU layer)
    - comm_energy_j   (upload size * energy per MB)
"""

import os
import sys
import torch
import numpy as np


def compute_energy(train_time, update):
    """
    Compute energy usage for:
        - compute (time)
        - compute FLOPs
        - communication (MB)
    """

    # Device profile
    watts = float(os.environ.get("DEVICE_POWER_WATTS", 3.5))
    joules_per_mb = float(os.environ.get("NET_J_PER_MB", 0.6))

    # Compute energy from time
    compute_time_j = watts * train_time

    # FLOPs (very rough estimate)
    flops = 1014.0
    compute_flops_j = flops * 1e-6

    # Communication cost
    total_bytes = 0
    for k, v in update.items():
        total_bytes += v.numpy().nbytes

    mb = total_bytes / (1024 * 1024)
    comm_energy_j = mb * joules_per_mb

    # Fake "bandwidth" estimation from upload time
    bandwidth_mbps = np.random.uniform(10, 50)

    total_energy_j = compute_time_j + compute_flops_j + comm_energy_j

    return {
        "compute_time_j": compute_time_j,
        "compute_flops_j": compute_flops_j,
        "comm_energy_j": comm_energy_j,
        "total_energy_j": total_energy_j,
        "bandwidth_mbps": bandwidth_mbps,
    }
