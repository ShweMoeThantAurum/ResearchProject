"""
Main client execution loop for federated learning.

Responsibilities:
 - Load local dataset partition and initialise model.
 - Download the global model each round.
 - Perform local training (FedAvg / AEFL / FedProx).
 - Optionally apply Differential Privacy and/or Compression.
 - Estimate round-level energy consumption.
 - Upload processed updates and metadata to the server.

This module contains no global state except environment variables and
is executed inside each client container.
"""

import os
import torch
import numpy as np

from src.models.simple_gru import SimpleGRU
from src.fl.logger import log_event
from src.fl.utils import (
    get_proc_dir,
    get_fl_rounds,
    get_batch_size,
    get_local_epochs,
    get_lr,
    get_hidden_size,
)

from src.fl.client.cleanup import cleanup_local_tmp
from src.fl.client.data import load_local_data
from src.fl.client.train import train_one_round
from src.fl.client.energy import estimate_round_energy
from src.fl.client.meta import build_round_metadata
from src.fl.client.modes import get_client_mode, client_allows_training
from src.fl.client.s3 import download_global, upload_processed_update, upload_metadata
from src.fl.client.privacy import maybe_add_dp_noise
from src.fl.client.compression import maybe_compress


# ============================================================================
# Local-epoch adaptation for AEFL
# ============================================================================
def get_effective_local_epochs(base_epochs: int, mode: str, role: str) -> int:
    """
    Decide the number of local training epochs for this client.

    AEFL reduces epochs for high-power roles (camera, bus) to save energy.
    FedAvg / FedProx use the configured base value.

    Returns:
        int: number of epochs for this round (≥0)
    """
    if base_epochs <= 0:
        return 0

    if mode != "aefl":
        return base_epochs

    # AEFL adaptive rule:
    # - camera, bus → heavier nodes → run ~half the epochs (minimum 1)
    if role in ("camera", "bus"):
        return max(1, base_epochs // 2)

    return base_epochs


# ============================================================================
# Main entrypoint for a client container
# ============================================================================
def main():
    """Run all federated learning rounds for this client."""
    role = os.environ.get("CLIENT_ROLE", "roadside")
    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = get_client_mode()  # "aefl", "fedavg", "fedprox"
    variant = os.environ.get("VARIANT_ID", "").strip()

    fl_rounds = get_fl_rounds()
    proc_dir = get_proc_dir()
    batch_size = get_batch_size()
    base_epochs = get_local_epochs()
    lr = get_lr()
    hidden_size = get_hidden_size()

    # Energy model parameters
    device_power_watts = float(os.environ.get("DEVICE_POWER_WATTS", "3.5"))
    net_j_per_mb = float(os.environ.get("NET_J_PER_MB", "0.6"))

    # DP + Compression flags
    dp_enabled = os.environ.get("DP_ENABLED", "false").lower() == "true"
    dp_sigma = float(os.environ.get("DP_SIGMA", "0.0"))
    compression_enabled = (
        os.environ.get("COMPRESSION_ENABLED", "false").lower() == "true"
    )
    compression_mode = os.environ.get("COMPRESSION_MODE", "").lower()

    # Determine graph width (num_nodes)
    x_train_path = os.path.join(proc_dir, "X_train.npy")
    if not os.path.exists(x_train_path):
        raise FileNotFoundError(
            "Missing X_train.npy — preprocessing required before training."
        )

    num_nodes = np.load(x_train_path).shape[-1]
    device = "cpu"

    print(
        f"[{role}] Client start | dataset={dataset}, mode={mode}, variant='{variant}', "
        f"nodes={num_nodes}, rounds={fl_rounds}, base_local_epochs={base_epochs}"
    )

    # Log configuration for reproducibility
    log_event(
        "client_config.log",
        {
            "role": role,
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "num_nodes": num_nodes,
            "rounds": fl_rounds,
            "batch_size": batch_size,
            "base_epochs": base_epochs,
            "lr": lr,
            "hidden_size": hidden_size,
            "device_power_watts": device_power_watts,
            "net_j_per_mb": net_j_per_mb,
            "dp_enabled": dp_enabled,
            "dp_sigma": dp_sigma,
            "compression_enabled": compression_enabled,
            "compression_mode": compression_mode,
        },
    )

    # ------------------------------------------------------------
    # Cleanup old temporary artefacts before starting the rounds
    # ------------------------------------------------------------
    cleanup_local_tmp(role)

    # Local data loader is reused across all rounds
    loader = load_local_data(
        proc_dir=proc_dir,
        role=role,
        num_nodes=num_nodes,
        batch_size=batch_size,
        local_epochs=base_epochs,
        lr=lr,
    )

    # Initialise local model
    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size).to(device)
    total_energy_j = 0.0

    # ============================================================================
    # Federated rounds
    # ============================================================================
    for r in range(1, fl_rounds + 1):
        print(f"\n[{role}] ===== ROUND {r} =====")

        # --------------------------------------------------------
        # 1) Download global model from server
        # --------------------------------------------------------
        global_path, dl_bytes = download_global(r, role)
        global_state = torch.load(global_path, map_location=device)
        model.load_state_dict(global_state)

        # --------------------------------------------------------
        # 2) Determine local epochs (adaptive for AEFL)
        # --------------------------------------------------------
        effective_epochs = get_effective_local_epochs(base_epochs, mode, role)
        print(
            f"[{role}] Round {r}: effective_local_epochs={effective_epochs} "
            f"(base={base_epochs})"
        )

        # --------------------------------------------------------
        # 3) Local training
        # --------------------------------------------------------
        if client_allows_training(mode) and effective_epochs > 0:
            prox_ref = (
                {k: v.clone().to(device) for k, v in global_state.items()}
                if mode == "fedprox"
                else None
            )

            updated_state, train_time, train_loss, train_samples = train_one_round(
                model=model,
                loader=loader,
                role=role,
                round_id=r,
                device=device,
                local_epochs=effective_epochs,
                lr=lr,
                mode=mode,
                global_state=prox_ref,
            )
        else:
            # No training → re-upload global state as-is
            updated_state = {k: v.cpu() for k, v in model.state_dict().items()}
            train_time, train_loss, train_samples = 0.0, 0.0, 0

        # --------------------------------------------------------
        # 4) Apply DP noise and/or Compression
        # --------------------------------------------------------
        dp_state = maybe_add_dp_noise(updated_state)
        comp_state, kept_ratio, modeled_bytes = maybe_compress(dp_state)

        log_event(
            "client_update_processing.log",
            {
                "role": role,
                "round": r,
                "dataset": dataset,
                "mode": mode,
                "variant": variant,
                "effective_epochs": effective_epochs,
                "dp_enabled": dp_enabled,
                "dp_sigma": dp_sigma,
                "compression_enabled": compression_enabled,
                "compression_mode": compression_mode,
                "compression_kept_ratio": kept_ratio,
                "compression_modeled_bytes": modeled_bytes,
            },
        )

        # --------------------------------------------------------
        # 5) Upload processed update
        # --------------------------------------------------------
        update_bytes, upload_latency = upload_processed_update(r, role, comp_state)

        # --------------------------------------------------------
        # 6) Energy estimation for this FL round
        # --------------------------------------------------------
        energy_record = estimate_round_energy(
            role=role,
            round_id=r,
            train_time_sec=train_time,
            download_bytes=dl_bytes,
            upload_bytes=update_bytes,
            device_power_watts=device_power_watts,
            net_j_per_mb=net_j_per_mb,
            num_nodes=num_nodes,
            hidden_size=hidden_size,
        )
        total_energy_j += float(energy_record["total_energy_j"])

        # --------------------------------------------------------
        # 7) Upload metadata (energy + bandwidth + training stats)
        # --------------------------------------------------------
        meta = build_round_metadata(
            role=role,
            round_id=r,
            energy_record=energy_record,
            train_loss=train_loss,
            train_samples=train_samples,
            update_bytes=update_bytes,
            upload_latency_sec=upload_latency,
        )
        upload_metadata(r, role, meta)

    # ============================================================================
    # Final energy summary for this client
    # ============================================================================
    log_event(
        "client_energy_summary.log",
        {
            "role": role,
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "rounds": fl_rounds,
            "total_energy_j": total_energy_j,
        },
    )

    print(
        f"[{role}] Completed {fl_rounds} rounds. "
        f"Total estimated energy = {total_energy_j:.2f} J."
    )


if __name__ == "__main__":
    main()
