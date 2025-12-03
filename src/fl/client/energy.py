"""
Energy estimation utilities for client-side training and communication.
Computes simple Joule estimates from time and payload size.
"""

def compute_compute_energy(duration_s, device_power_watts):
    """Energy from local computation using E = P * t."""
    return device_power_watts * duration_s


def compute_comm_energy(num_bytes, net_j_per_mb):
    """Energy from communication, proportional to MB transferred."""
    mb = num_bytes / (1024.0 * 1024.0)
    return mb * net_j_per_mb


def compute_energy(compute_duration_s,
                   update_size_bytes,
                   device_power_watts,
                   net_j_per_mb):
    """
    Compute total round energy as compute + communication.
    """
    e_comp = compute_compute_energy(compute_duration_s, device_power_watts)
    e_comm = compute_comm_energy(update_size_bytes, net_j_per_mb)
    total = e_comp + e_comm

    return {
        "compute_energy": e_comp,
        "comm_energy": e_comm,
        "total_energy": total,
    }
