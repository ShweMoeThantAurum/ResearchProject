"""
Energy estimation utilities for client rounds.

Each round produces a JSON entry (energy_<role>.jsonl) containing:
 - compute energy (device power × training time)
 - communication energy (bytes × J/MB)
 - FLOP-based estimate (optional, logged only)
"""

import os
import json
from pathlib import Path
from src.utils.flops import estimate_gru_flops

ENERGY_PER_FLOP_J = 1e-9  # academic assumption


def _log_energy(role, record):
    """Append a JSON record for this round to run_logs/."""
    path = Path("run_logs")
    path.mkdir(exist_ok=True)
    with open(path / f"energy_{role}.jsonl", "a") as f:
        f.write(json.dumps(record) + "\n")


def estimate_round_energy(
    role,
    round_id,
    train_time_sec,
    download_bytes,
    upload_bytes,
    device_power_watts,
    net_j_per_mb,
    num_nodes=None,
    hidden_size=None,
    seq_len=12,
):
    """
    Estimate compute + communication energy for a client round.
    """
    dataset = os.environ.get("DATASET", "unknown")
    mode = os.environ.get("FL_MODE", "aefl").lower()
    variant = os.environ.get("VARIANT_ID", "")

    compute_j = device_power_watts * train_time_sec

    # Optional FLOPs energy
    if num_nodes and hidden_size:
        flops = estimate_gru_flops(num_nodes, hidden_size, seq_len)
        compute_flops_j = flops * ENERGY_PER_FLOP_J
    else:
        compute_flops_j = 0.0

    d_mb = download_bytes / (1024 * 1024)
    u_mb = upload_bytes / (1024 * 1024)
    comm_j = net_j_per_mb * (d_mb + u_mb)

    total = compute_j + comm_j

    record = {
        "dataset": dataset.lower(),
        "mode": mode,
        "variant": variant,
        "role": role,
        "round": round_id,
        "compute_j_time": compute_j,
        "compute_j_flops": compute_flops_j,
        "comm_j_total": comm_j,
        "total_energy_j": total,
    }

    _log_energy(role, record)

    print(
        f"[{role}] Energy r={round_id}: compute={compute_j:.4f} J, "
        f"comm={comm_j:.4f} J, TOTAL={total:.4f} J"
    )

    return record
