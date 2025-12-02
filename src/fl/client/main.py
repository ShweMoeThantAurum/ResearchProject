"""
Main client loop for cloud-based federated learning.

Each client downloads the global model, trains locally, applies optional
DP and compression, uploads the processed update, logs energy usage, and
stores metadata for server-side adaptive selection.
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

from src.fl.client.s3 import (
    download_global,
    upload_processed_update,
    upload_metadata,
)

from src.fl.client.privacy import maybe_add_dp_noise
from src.fl.client.compression import maybe_compress


def main():
    """Run the client training sequence for all federated rounds."""
    role = os.environ.get("CLIENT_ROLE", "roadside")

    proc_dir = get_proc_dir()
    fl_rounds = get_fl_rounds()
    batch_size = get_batch_size()
    local_epochs = get_local_epochs()
    lr = get_lr()
    hidden_size = get_hidden_size()

    device_power_watts = float(os.environ.get("DEVICE_POWER_WATTS", "3.5"))
    net_j_per_mb = float(os.environ.get("NET_J_PER_MB", "0.6"))

    device = "cpu"

    # Infer graph dimensionality
    x_train_path = os.path.join(proc_dir, "X_train.npy")
    if not os.path.exists(x_train_path):
        raise FileNotFoundError(f"Missing {x_train_path}. Did you run preprocessing?")
    num_nodes = np.load(x_train_path).shape[-1]

    mode = get_client_mode()
    print(
        f"[{role}] STARTING cloud client | mode={mode} | "
        f"nodes={num_nodes}, rounds={fl_rounds}"
    )

    cleanup_local_tmp(role)

    loader = load_local_data(
        proc_dir=proc_dir,
        role=role,
        num_nodes=num_nodes,
        batch_size=batch_size,
        local_epochs=local_epochs,
        lr=lr,
    )

    model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden_size).to(device)
    total_energy_j = 0.0

    print(f"[{role}] Lambda offload disabled (local only)")

    # Federated rounds
    for r in range(1, fl_rounds + 1):
        print(f"\n[{role}] ===== ROUND {r} =====")

        # Download global model
        global_path, dl_bytes = download_global(r, role)
        global_state = torch.load(global_path, map_location=device)
        model.load_state_dict(global_state)

        # Train locally
        if not client_allows_training(mode):
            updated_state = {k: v.cpu() for k, v in model.state_dict().items()}
            train_time = 0.0
            train_loss = 0.0
            train_samples = 0
        else:
            prox_ref_state = (
                {k: v.clone().to(device) for k, v in global_state.items()}
                if mode.lower() == "fedprox"
                else None
            )

            updated_state, train_time, train_loss, train_samples = train_one_round(
                model=model,
                loader=loader,
                role=role,
                round_id=r,
                device=device,
                local_epochs=local_epochs,
                lr=lr,
                mode=mode,
                global_state=prox_ref_state,
            )

        # DP noise → Compression
        dp_state = maybe_add_dp_noise(updated_state)
        comp_state, kept_ratio, modeled_bytes = maybe_compress(dp_state)

        processed_bytes, up_latency = upload_processed_update(r, role, comp_state)

        # Energy estimation
        energy_record = estimate_round_energy(
            role=role,
            round_id=r,
            train_time_sec=train_time,
            download_bytes=dl_bytes,
            upload_bytes=processed_bytes,
            device_power_watts=device_power_watts,
            net_j_per_mb=net_j_per_mb,
            num_nodes=num_nodes,
            hidden_size=hidden_size,
            seq_len=12,
        )


        total_energy_j += energy_record["total_energy_j"]


        # Metadata upload
        meta = build_round_metadata(
            role=role,
            round_id=r,
            energy_record=energy_record,
            train_loss=train_loss,
            train_samples=train_samples,
            update_bytes=processed_bytes,
            upload_latency_sec=up_latency,
        )
        upload_metadata(r, role, meta)

    # Final summary
    log_event(
        "client_energy_summary.log",
        {
            "role": role,
            "rounds": fl_rounds,
            "total_energy_j": total_energy_j,
            "mode": mode,
            "lambda_offload": False,
        },
    )

    print(
        f"[{role}] Finished {fl_rounds} rounds. "
        f"Total estimated energy={total_energy_j:.2f} J."
    )


if __name__ == "__main__":
    main()
