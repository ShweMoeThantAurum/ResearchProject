"""
Energy estimation utilities for FL clients.
Models compute cost (time-based) and communication cost (bytes-based).
"""


def compute_compute_energy(duration_s, device_power_watts):
    """
    Compute energy used for local computation:
        E = P * t  (Joules)
    """
    return device_power_watts * duration_s


def compute_comm_energy(num_bytes, net_j_per_mb):
    """
    Compute energy for wireless transmission:
        E_comm = size(MB) * J_per_MB
    """
    mb = num_bytes / (1024.0 * 1024.0)
    return mb * net_j_per_mb


def compute_energy(compute_duration_s, update_size_bytes, device_power_watts, net_j_per_mb):
    """
    Main helper used by client_main.

    Args:
        compute_duration_s: local training time in seconds
        update_size_bytes: size of uploaded update in bytes
        device_power_watts: device power draw (W)
        net_j_per_mb: energy per MB transmitted (J/MB)

    Returns:
        dict with compute_j, comm_j, total_j
    """
    e_comp = compute_compute_energy(compute_duration_s, device_power_watts)
    e_comm = compute_comm_energy(update_size_bytes, net_j_per_mb)
    total = e_comp + e_comm

    return {
        "compute_j": e_comp,
        "comm_j": e_comm,
        "total_j": total,
    }
