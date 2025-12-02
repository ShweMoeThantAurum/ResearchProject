"""
Energy estimation for client compute and communication.
"""

from src.fl.config import settings


def compute_energy_round(train_time, flops_j, comm_mb):
    """
    Estimate energy usage for one round.
    """
    power = settings.get_device_power()
    comm_j_per_mb = settings.get_comm_energy_per_mb()

    compute = power * train_time
    flops_energy = flops_j
    comm = comm_mb * comm_j_per_mb

    return compute + flops_energy + comm
