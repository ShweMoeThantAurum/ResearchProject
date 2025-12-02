"""
Energy estimation utilities for federated learning clients.

This module combines:
    - Compute energy from training time and device power
    - Optional extra compute cost from floating point operations
    - Communication energy from upload and download volume
"""

from src.fl.utils.logger import log_event


def estimate_round_energy(role,
                          round_id,
                          train_time_sec,
                          train_flops,
                          download_bytes,
                          upload_bytes,
                          device_power_watts,
                          net_j_per_mb,
                          flop_energy_j):
    """
    Estimate compute and communication energy for a single round.

    Parameters:
        role               : client role name
        round_id           : federated round index
        train_time_sec     : local training time in seconds
        train_flops        : approximate floating point operations
        download_bytes     : bytes downloaded (global model)
        upload_bytes       : bytes uploaded (client update)
        device_power_watts : effective device power during training
        net_j_per_mb       : joules per megabyte for communication
        flop_energy_j      : joules per floating point operation

    Returns:
        record: dictionary with detailed energy statistics
    """
    compute_time_j = device_power_watts * train_time_sec
    compute_flops_j = train_flops * flop_energy_j

    total_bytes = download_bytes + upload_bytes
    mb = total_bytes / (1024.0 * 1024.0)
    comm_j = net_j_per_mb * mb

    total_j = compute_time_j + compute_flops_j + comm_j

    record = {
        "role": role,
        "round": round_id,
        "train_time_sec": train_time_sec,
        "train_flops": train_flops,
        "download_bytes": download_bytes,
        "upload_bytes": upload_bytes,
        "download_mb": download_bytes / (1024.0 * 1024.0),
        "upload_mb": upload_bytes / (1024.0 * 1024.0),
        "compute_time_j": compute_time_j,
        "compute_flops_j": compute_flops_j,
        "comm_j": comm_j,
        "total_j": total_j,
    }

    log_event("client_energy.log", record)

    print("[%s] Energy round %d: compute_time=%.2f J, compute_flops=%.6f J, comm_total=%.4f J, total=%.4f J"
          % (role, round_id, compute_time_j, compute_flops_j, comm_j, total_j))

    return record


def approximate_flops_per_round(total_samples,
                                num_nodes,
                                hidden_size,
                                seq_len):
    """
    Approximate the number of floating point operations for one round.

    This uses a very rough closed form for a single-layer GRU followed
    by a linear layer, multiplied by the number of samples.

    The goal is to capture relative differences across modes and datasets
    rather than exact hardware-level counts.
    """
    # Rough GRU cost per timestep per sample:
    # three gates with input and hidden projections
    # cost is proportional to:
    #   3 * (num_nodes * hidden_size) for input projections
    #   3 * (hidden_size * hidden_size) for hidden projections
    gru_per_step = 3.0 * (num_nodes * hidden_size + hidden_size * hidden_size)

    # Linear layer from hidden state to nodes
    linear_cost = hidden_size * num_nodes

    flops_per_sample = seq_len * gru_per_step + linear_cost

    total_flops = flops_per_sample * float(total_samples)
    return total_flops
