"""
Client-side main loop for federated learning.

Responsibilities:
    - Load global model from S3 for each round
    - Train locally on client-specific partitions
    - Compute energy consumption (compute + communication)
    - Apply optional:
        • differential privacy (DP)
        • gradient compression
    - Upload processed model update and metadata
"""

import os
import time
import json
import torch

from src.fl.client.train import train_one_round
from src.fl.client.energy import estimate_energy
from src.fl.client.comm import (
    download_global_model,
    upload_client_update,
    upload_client_metadata,
)
from src.fl.client.dp import apply_dp
from src.fl.client.compression import apply_compression
from src.fl.client.utils_client import load_client_data

from src.fl.models import SimpleGRU


def main():
    """
    Entry point for the client container.
    Handles the full FL lifecycle for the assigned role.
    """
    role = os.environ.get("CLIENT_ROLE")
    dataset = os.environ.get("DATASET")
    mode = os.environ.get("FL_MODE", "aefl").lower()
    rounds = int(os.environ.get("FL_ROUNDS", 20))
    hidden = int(os.environ.get("HIDDEN_SIZE", 64))
    device = torch.device("cpu")

    proc_dir = os.path.join("datasets", "processed", dataset)
    bucket = os.environ.get("S3_BUCKET", "aefl")

    if role is None:
        raise RuntimeError("CLIENT_ROLE not set in environment")

    print("[%s] Starting federated client | dataset=%s mode=%s" %
          (role, dataset, mode))

    # Load local dataset partition
    X_local, y_local = load_client_data(proc_dir, role)
    num_nodes = X_local.shape[-1]

    # Prepare model
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden).to(device)

    for r in range(1, rounds + 1):
        print("\n[%s] ===== ROUND %d =====" % (role, r))

        # Download global model
        print("[%s] Waiting for global model round %d..." % (role, r))
        state = download_global_model(bucket, dataset, mode, r, timeout=600)

        model.load_state_dict(state)

        # Local training
        loss, train_time, flops = train_one_round(
            model=model,
            X_local=X_local,
            y_local=y_local,
            lr=float(os.environ.get("LR", 0.001)),
            batch_size=int(os.environ.get("BATCH_SIZE", 64)),
            epochs=int(os.environ.get("LOCAL_EPOCHS", 1)),
            device=device,
        )

        print("[%s] Round %d training | loss=%.6f time=%.3fs" %
              (role, r, loss, train_time))

        # Extract update
        updated_state = model.state_dict()

        # Optional: Differential Privacy
        if os.environ.get("DP_ENABLED", "false").lower() == "true":
            sigma = float(os.environ.get("DP_SIGMA", 0.01))
            apply_dp(updated_state, sigma)

        # Optional: Compression
        if os.environ.get("COMPRESSION_ENABLED", "false").lower() == "true":
            mode_c = os.environ.get("COMPRESSION_MODE", "sparsify")
            sparsity = float(os.environ.get("COMPRESSION_SPARSITY", 0.5))
            k_frac = float(os.environ.get("COMPRESSION_K_FRAC", 0.1))

            apply_compression(
                updated_state,
                mode_c,
                sparsity=sparsity,
                k_frac=k_frac,
            )

        # Estimate energy
        energy = estimate_energy(
            train_time=train_time,
            flops=flops,
            update_size_mb=0.204,  # constant for now
            power_watts=float(os.environ.get("DEVICE_POWER_WATTS", 3.5)),
            net_j_per_mb=float(os.environ.get("NET_J_PER_MB", 0.6)),
        )

        # Upload update
        upload_client_update(
            bucket=bucket,
            dataset=dataset,
            mode=mode,
            round_id=r,
            role=role,
            state_dict=updated_state,
        )

        # Upload metadata
        meta = {
            "bandwidth_mbps": energy["bw"],
            "total_energy_j": energy["total"],
            "compute_j": energy["compute"],
            "comm_j": energy["comm"],
        }

        upload_client_metadata(
            bucket=bucket,
            dataset=dataset,
            mode=mode,
            round_id=r,
            role=role,
            metadata=meta,
        )

        print("[%s] Energy round %d: total=%.3f J" %
              (role, r, energy["total"]))

    print("[%s] Finished %d rounds." % (role, rounds))


if __name__ == "__main__":
    main()
