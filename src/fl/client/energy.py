"""
Energy estimation utilities for client-side training.

Estimates:
- computation energy from device power and time
- communication energy from update size and link cost.
"""


def compute_compute_energy(duration_s, device_power_watts):
    """Compute local computation energy in Joules."""
    return device_power_watts * duration_s


def compute_comm_energy(num_bytes, net_j_per_mb):
    """Compute communication energy from payload size in bytes."""
    mb = num_bytes / (1024.0 * 1024.0)
    return mb * net_j_per_mb


def compute_energy(compute_duration_s, update_size_bytes, device_power_watts, net_j_per_mb):
    """Compute compute, communication, and total energy for one round."""
    e_comp = compute_compute_energy(compute_duration_s, device_power_watts)
    e_comm = compute_comm_energy(update_size_bytes, net_j_per_mb)
    total = e_comp + e_comm
    return e_comp, e_comm, total
