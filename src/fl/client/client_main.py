"""
Main client loop for federated learning.
Downloads global model, trains locally, applies DP and compression,
uploads updates and metadata, and logs energy per round.
"""

import time
import torch
import numpy as np

from ..models.gru_model import GRUModel
from ..utils.logger import log_event
from ..data.loader import load_client_loader
from .utils_client import (
    get_role,
    get_dataset,
    get_fl_mode,
    get_fl_rounds,
    get_batch_size,
    get_local_epochs,
    get_lr,
    get_hidden_size,
    build_round_metadata,
    cleanup_local_tmp,
)
from .train import train_one_round
from .energy import estimate_round_energy
from .comm import download_global_model, upload_update, upload_metadata
from .dp import apply_dp_if_enabled
from .compression import apply_compression_if_enabled


def main():
    """Run the full client training sequence for all rounds."""
    role = get_role()
    dataset = get_dataset()
    mode = get_fl_mode()
    rounds = get_fl_rounds()
    batch_size = get_batch_size()
    local_epochs = get_local_epochs()
    lr = get_lr()
    hidden_size = get_hidden_size()

    device = "cpu"

    print(
        f"[{role}] Starting client | dataset={dataset}, mode={mode}, "
        f"rounds={rounds}, batch={batch_size}, epochs={local_epochs}, lr={lr}"
    )

    cleanup_local_tmp(role)

    # Load local data once to infer num_nodes and build DataLoader
    loader, num_nodes = load_client_loader(dataset, role, batch_size)

    model = GRUModel(num_nodes=num_nodes, hidden_size=hidden_size).to(device)

    total_energy_j = 0.0

    for r in range(1, rounds + 1):
        print(f"\n[{role}] ===== ROUND {r} =====")

        # Download global model for this round
        global_path, dl_bytes = download_global_model(r, role)
        global_state = torch.load(global_path, map_location=device)
        model.load_state_dict(global_state)

        # Local training
        train_start = time.time()

        if mode.lower() == "localonly":
            updated_state = {k: v.cpu() for k, v in model.state_dict().items()}
            train_time_sec = 0.0
            avg_loss = 0.0
            train_samples = 0
            approx_flops = 0.0
        else:
            prox_ref_state = None
            if mode.lower() == "fedprox":
                prox_ref_state = {k: v.clone().to(device) for k, v in global_state.items()}

            updated_state, avg_loss, train_samples, approx_flops = train_one_round(
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
            train_time_sec = time.time() - train_start

        # DP
        dp_state = apply_dp_if_enabled(updated_state, role, r)

        # Compression
        comp_state, kept_ratio, modeled_bytes = apply_compression_if_enabled(dp_state, role, r)

        # Upload processed update
        up_bytes, up_latency = upload_update(r, role, comp_state)

        # Energy estimation based on actual bytes
        energy_record = estimate_round_energy(
            role=role,
            round_id=r,
            train_time_sec=train_time_sec,
            approx_flops=approx_flops,
            download_bytes=dl_bytes,
            upload_bytes=up_bytes,
        )

        total_energy_j += energy_record["total_j"]

        # Metadata for AEFL selection
        meta = build_round_metadata(
            role=role,
            round_id=r,
            train_loss=avg_loss,
            train_samples=train_samples,
            compute_time_j=energy_record["compute_time_j"],
            compute_flops_j=energy_record["compute_flops_j"],
            comm_j=energy_record["comm_j"],
            download_bytes=dl_bytes,
            upload_bytes=up_bytes,
            upload_latency_sec=up_latency,
        )
        upload_metadata(r, role, meta)

    log_event({
        "type": "client_summary",
        "role": role,
        "rounds": rounds,
        "total_energy_j": total_energy_j,
        "mode": mode,
        "dataset": dataset,
    })

    print(
        f"[{role}] Finished {rounds} rounds. "
        f"Total estimated energy={total_energy_j:.2f} J."
    )
