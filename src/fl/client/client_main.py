"""
Client-side federated learning loop.
Loads local dataset, trains GRU model, computes energy, applies DP/compression,
uploads update + metadata to S3 for each round.
"""

import os
import time
import torch
import json

from .train import train_one_epoch
from .energy import estimate_energy
from .comm import upload_update, download_global_model
from .utils_client import load_client_partition
from ..models.gru_model import GRUModel
from ..config.settings import load_settings
from ..utils.logger import log_event


def main():
    """Entry point for each FL client."""
    settings = load_settings()

    role = settings.role
    dataset = settings.dataset
    rounds = settings.rounds

    print(f"[{role}] Starting client | dataset={dataset} mode={settings.mode} rounds={rounds} device={settings.device}")

    # --------------------------------------------------
    # 1. Load local partition
    # --------------------------------------------------
    X, y = load_client_partition(dataset, role)
    num_nodes = X.shape[-1]

    # --------------------------------------------------
    # 2. Create model
    # --------------------------------------------------
    model = GRUModel(num_nodes=num_nodes, hidden_size=settings.hidden_size)
    model.to(settings.device)

    # --------------------------------------------------
    # 3. Per-round FL loop
    # --------------------------------------------------
    total_energy = 0.0

    for r in range(1, rounds + 1):
        print(f"[{role}] ===== ROUND {r} =====")

        # ----------------------------------------------
        # Download global model from S3
        # ----------------------------------------------
        global_state = download_global_model(r)
        if global_state is None:
            print(f"[{role}] Waiting for global model round {r}...")
            time.sleep(2)
            continue

        model.load_state_dict(global_state)

        # ----------------------------------------------
        # Local training
        # ----------------------------------------------
        loss, train_time = train_one_epoch(
            model,
            X,
            y,
            lr=settings.lr,
            batch_size=settings.batch_size,
            local_epochs=settings.local_epochs,
            device=settings.device,
            dp_enabled=settings.dp_enabled,
            dp_sigma=settings.dp_sigma,
            compression_enabled=settings.compression_enabled,
            compression_mode=settings.compression_mode,
            compression_sparsity=settings.compression_sparsity,
            compression_k_frac=settings.compression_k_frac,
        )

        print(f"[{role}] Round {r} training | loss={loss:.6f}, time={train_time:.3f}s, samples={len(X)}")

        # ----------------------------------------------
        # Compute energy
        # ----------------------------------------------
        energy_info = estimate_energy(
            compute_time=train_time,
            update_size_mb=None,   # filled after upload
            device_watts=settings.device_power_watts,
            j_per_mb=settings.net_j_per_mb,
        )

        # ----------------------------------------------
        # Upload update to S3
        # ----------------------------------------------
        update_state = model.state_dict()

        update_size_mb, latency = upload_update(r, role, update_state)

        energy_info["comm_total"] = update_size_mb * settings.net_j_per_mb
        energy_info["total"] = energy_info["compute_time"] + energy_info["comm_total"]

        total_energy += energy_info["total"]

        print(f"[{role}] Energy round {r}: compute={energy_info['compute_time']:.2f} J, "
              f"comm={energy_info['comm_total']:.2f} J, total={energy_info['total']:.2f} J")

        # ----------------------------------------------
        # Upload metadata
        # ----------------------------------------------
        metadata = {
            "bandwidth_mbps": (update_size_mb * 8) / (latency + 1e-9),
            "total_energy_j": energy_info["total"],
        }

        upload_update(r, role, metadata, meta=True)

    print(f"[{role}] Finished {rounds} rounds. Total estimated energy={total_energy:.2f} J.")


if __name__ == "__main__":
    main()
