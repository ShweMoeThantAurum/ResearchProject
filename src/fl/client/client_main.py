"""
Client driver for federated learning.
Downloads global model, performs local training,
applies DP/compression, computes energy, and uploads results.
"""

import time
import torch

from src.fl.config.settings import settings
from src.fl.data.loader import load_client_partition
from src.fl.models.gru_model import GRUModel

from src.fl.client.train import train_one_round
from src.fl.client.energy import compute_energy
from src.fl.client.dp import apply_dp_noise
from src.fl.client.compression import compress_update
from src.fl.client.comm import (
    download_global_model,
    upload_update,
    upload_metadata,
)

from src.fl.utils.logger import log_event


def main():
    """Entry point for each federated client."""
    role = settings.client_role
    dataset = settings.dataset
    rounds = settings.fl_rounds

    log_event(f"[{role}] Starting client for dataset={dataset}")

    # Load local partition
    loader, num_nodes = load_client_partition(dataset, role, settings.batch_size)

    # Initial model
    model = GRUModel(num_nodes=num_nodes, hidden_size=settings.hidden_size)

    device = torch.device("cpu")  # GPU future work

    total_energy = 0.0

    for r in range(1, rounds + 1):
        print(f"[{role}] ===== ROUND {r} =====")

        # Download global model
        state = download_global_model(r, role)
        model.load_state_dict(state)

        # Train
        train_start = time.time()
        updated, avg_loss, samples, flops = train_one_round(
            model=model,
            loader=loader,
            role=role,
            round_id=r,
            device=device,
            local_epochs=settings.local_epochs,
            lr=settings.lr,
            mode=settings.fl_mode,
            global_state=state,
        )
        train_time = time.time() - train_start

        # Apply DP
        if settings.dp_enabled:
            updated = apply_dp_noise(updated, sigma=settings.dp_sigma)

        # Apply compression
        if settings.compression_enabled:
            updated = compress_update(updated, settings.compression_sparsity)

        # Compute update size
        update_size_bytes = sum(v.nelement() * v.element_size() for v in updated.values())

        # Compute energy
        e = compute_energy(
            compute_duration_s=train_time,
            update_size_bytes=update_size_bytes,
            device_power_watts=settings.device_power_watts,
            net_j_per_mb=settings.net_j_per_mb,
        )
        total_energy += e["total"]

        # Upload update
        upload_update(r, role, updated)

        # Upload metadata
        upload_metadata(r, role, {
            "total_energy_j": e["total"],
            "compute_energy_j": e["compute_energy"],
            "comm_energy_j": e["comm_energy"],
            "num_samples": samples,
        })

        print(
            f"[{role}] Round {r}: loss={avg_loss:.6f} "
            f"time={train_time:.2f}s energy={e['total']:.3f}J"
        )

    print(f"[{role}] Finished {rounds} rounds. Total estimated energy={total_energy:.2f} J.")
