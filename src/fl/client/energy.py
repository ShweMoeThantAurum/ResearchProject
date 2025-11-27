"""Client-side energy estimation for communication and computation."""

from src.fl.logger import log_event


def estimate_round_energy(role: str,
                          round_id: int,
                          train_time_sec: float,
                          download_bytes: int,
                          upload_bytes: int,
                          device_power_watts: float,
                          net_j_per_mb: float):
    """
    Estimate energy consumption for one FL round.

    Uses:
      - compute energy = power * time
      - communication energy = J/MB * transferred_MB
    """
    compute_j = device_power_watts * train_time_sec

    total_bytes = download_bytes + upload_bytes
    mb = total_bytes / (1024 * 1024)
    comm_j = net_j_per_mb * mb

    total_j = compute_j + comm_j

    record = {
        "role": role,
        "round": round_id,
        "train_time_sec": train_time_sec,
        "download_bytes": download_bytes,
        "upload_bytes": upload_bytes,
        "download_mb": download_bytes / (1024 * 1024),
        "upload_mb": upload_bytes / (1024 * 1024),
        "compute_j": compute_j,
        "comm_j": comm_j,
        "total_j": total_j,
    }

    log_event("client_energy.log", record)

    print(f"[{role}] Energy round {round_id}: compute={compute_j:.2f} J, "
          f"comm={comm_j:.2f} J, total={total_j:.2f} J")

    return record
