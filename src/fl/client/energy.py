"""
Energy estimation utilities for client devices.

We track:
    - compute energy (proportional to execution time)
    - communication energy (uploading update to server)
"""

import random


def estimate_energy(train_time,
                    flops,
                    update_size_mb,
                    power_watts,
                    net_j_per_mb):
    """
    Estimate compute and communication energy for a client round.

    compute_j = train_time * power_watts
    comm_j    = update_size_mb * net_j_per_mb
    total     = compute_j + comm_j

    We also generate a random bandwidth estimate for AEFL scoring.
    """
    compute = train_time * power_watts
    comm = update_size_mb * net_j_per_mb
    total = compute + comm

    # Simulated bandwidth: this only affects AEFL scoring
    bw = random.uniform(10.0, 55.0)

    return {
        "compute": compute,
        "comm": comm,
        "total": total,
        "bw": bw,
    }
