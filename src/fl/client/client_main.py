"""
Client-side federated learning loop.
Loads local dataset, trains GRU model, computes energy, applies DP/compression,
uploads update + metadata to S3 for each round.
"""

import time
import torch

from src.fl.config.settings import settings
from src.fl.config.config_loader import load_experiment_config
from src.fl.data.loader import load_client_partition
from src.fl.models.gru_model import GRUModel
from src.fl.client.train import train_one_epoch
from src.fl.client.energy import compute_energy
from src.fl.client.comm import (
    download_global_model,
    upload_update,
    upload_metadata,
)
from src.fl.utils.logger import log_event
from src.fl.client.utils_client import get_role, get_dataset
from src.fl.client.utils_client import compute_bandwidth_mbps


def main():
    """Entry point for each FL client."""
    # --------------------------------------------------
    # Load YAML configuration (baseline + optional overlays)
    # --------------------------------------------------
    load_experiment_config()

    # Role / dataset from env (with settings as fallback)
    role = settings.client_role or get_role()
    dataset = settings.dataset or get_dataset()
    rounds = settings.fl_rounds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(
        f"[{role}] Starting client | "
        f"dataset={dataset} mode={settings.fl_mode} rounds={rounds} device={device}"
    )
    log_event(
        f"[{role}] client_start dataset={dataset} mode={settings.fl_mode} rounds={rounds}"
    )

    # --------------------------------------------------
    # 1. Load local partition (tensors)
    # --------------------------------------------------
    X, y = load_client_partition(dataset, role)
    num_nodes = X.shape[-1]

    # --------------------------------------------------
    # 2. Create model
    # --------------------------------------------------
    model = GRUModel(num_nodes=num_nodes, hidden_size=settings.hidden_size).to(device)

    total_energy_j = 0.0

    # --------------------------------------------------
    # 3. Per-round FL loop
    # --------------------------------------------------
    for r in range(1, rounds + 1):
        print(f"[{role}] ===== ROUND {r} =====")

        # ----------------------------------------------
        # Download global model from S3
        # ----------------------------------------------
        global_state = download_global_model(r, role)
        # Safely load global state; if invalid, keep local weights
        try:
            model.load_state_dict(global_state)
        except Exception as e:
            print(
                f"[{role}] WARNING: Invalid global state for round {r} "
                f"(type={type(global_state)}): {e}"
            )
            log_event(
                f"[{role}] invalid_global_state round={r} "
                f"type={type(global_state)} error={e}"
            )
            # Continue with existing model weights (local state)

        # ----------------------------------------------
        # Local training + DP + compression
        # ----------------------------------------------
        update_state, loss, compute_time_s, batch_count = train_one_epoch(
            model=model,
            X=X,
            y=y,
            lr=settings.lr,
            batch_size=settings.batch_size,
            local_epochs=settings.local_epochs,
            device=device,
            dp_enabled=settings.dp_enabled,
            dp_sigma=settings.dp_sigma,
            compression_enabled=settings.compression_enabled,
            compression_mode=settings.compression_mode,
            compression_sparsity=settings.compression_sparsity,
            compression_k_frac=settings.compression_k_frac,
        )

        print(
            f"[{role}] Round {r} training | "
            f"loss={loss:.6f}, time={compute_time_s:.3f}s, batches={batch_count}"
        )
        log_event(
            f"[{role}] train_round r={r} "
            f"loss={loss:.6f} time_s={compute_time_s:.3f} batches={batch_count}"
        )

        # ----------------------------------------------
        # Upload update to S3 and measure size/latency
        # ----------------------------------------------
        size_bytes, upload_latency = upload_update(r, role, update_state)

        # ----------------------------------------------
        # Energy computation
        # ----------------------------------------------
        energy_info = compute_energy(
            compute_duration_s=compute_time_s,
            update_size_bytes=size_bytes,
            device_power_watts=settings.device_power_watts,
            net_j_per_mb=settings.net_j_per_mb,
        )

        total_energy_j += energy_info["total_j"]

        print(
            f"[{role}] Energy round {r}: "
            f"compute={energy_info['compute_j']:.2f} J, "
            f"comm={energy_info['comm_j']:.2f} J, "
            f"total={energy_info['total_j']:.2f} J"
        )
        log_event(
            f"[{role}] energy_round r={r} "
            f"compute_j={energy_info['compute_j']:.3f} "
            f"comm_j={energy_info['comm_j']:.3f} "
            f"total_j={energy_info['total_j']:.3f}"
        )

        # ----------------------------------------------
        # Upload metadata for AEFL selection
        # ----------------------------------------------
        bw_mbps = compute_bandwidth_mbps(size_bytes, upload_latency)
        metadata = {
            "bandwidth_mbps": float(bw_mbps),
            "total_energy_j": float(energy_info["total_j"]),
        }
        upload_metadata(r, role, metadata)

    print(
        f"[{role}] Finished {rounds} rounds. "
        f"Total estimated energy={total_energy_j:.2f} J."
    )
    log_event(
        f"[{role}] client_finished rounds={rounds} total_energy_j={total_energy_j:.3f}"
    )


if __name__ == "__main__":
    main()
