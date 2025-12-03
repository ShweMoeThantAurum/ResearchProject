"""
Client entrypoint for federated learning.

Each client:
- loads its local dataset partition
- receives global models from the server
- trains locally on CPU
- applies optional DP and compression
- computes simple energy estimates
- uploads updates and metadata to S3.
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
from src.fl.client.utils_client import (
    get_role,
    get_dataset,
    build_round_metadata,
)
from src.fl.utils.logger import log_event


def main():
    """Run one client process for all FL rounds."""
    role = get_role()
    dataset = get_dataset()
    rounds = settings.fl_rounds
    mode = settings.fl_mode

    # CPU-only training for AWS Academy; GPU can be future work.
    device = torch.device("cpu")

    print(
        f"[{role}] Starting client | dataset={dataset} "
        f"mode={mode} rounds={rounds} device={device}"
    )
    log_event(
        f"[{role}] start_client dataset={dataset} mode={mode} "
        f"rounds={rounds} device={device}"
    )

    loader = load_client_partition(dataset, role, batch_size=settings.batch_size)

    model = GRUModel(hidden_size=settings.hidden_size)
    total_energy = 0.0

    for r in range(1, rounds + 1):
        print(f"[{role}] ===== ROUND {r} =====")

        # Download latest global model from S3
        global_state = download_global_model(r, role)
        model.load_state_dict(global_state)

        # Local training
        start_train = time.time()
        updated_state, loss, samples, approx_flops = train_one_round(
            model=model,
            loader=loader,
            role=role,
            round_id=r,
            device=device,
            local_epochs=settings.local_epochs,
            lr=settings.lr,
            mode=mode,
            global_state=global_state,
        )
        train_time = time.time() - start_train

        # Apply differential privacy if enabled
        if settings.dp_enabled:
            updated_state = apply_dp_noise(updated_state, sigma=settings.dp_sigma)

        # Apply compression if enabled
        if settings.compression_enabled:
            updated_state = compress_update(updated_state, settings)

        # Upload update and measure communication cost
        size_bytes, upload_latency = upload_update(r, role, updated_state)

        # Energy accounting
        energy_info = compute_energy(
            compute_duration_s=train_time,
            update_size_bytes=size_bytes,
            device_power_watts=settings.device_power_watts,
            net_j_per_mb=settings.net_j_per_mb,
        )
        total_energy += energy_info["total_energy"]

        # Build and upload metadata for AEFL selection
        meta = build_round_metadata(
            role=role,
            round_id=r,
            train_loss=loss,
            train_samples=samples,
            compute_time_j=energy_info["compute_energy"],
            compute_flops_j=0.0,     # FLOPs-to-J conversion left as future refinement
            comm_j=energy_info["comm_energy"],
            download_bytes=0,        # Download size is not logged here
            upload_bytes=size_bytes,
            upload_latency_sec=upload_latency,
        )
        upload_metadata(r, role, meta)

        print(
            f"[{role}] Round {r} | loss={loss:.6f}, "
            f"time={train_time:.3f}s, "
            f"energy={energy_info['total_energy']:.3f} J"
        )

    print(
        f"[{role}] Finished {rounds} rounds. "
        f"Total estimated energy={total_energy:.2f} J."
    )
    log_event(
        f"[{role}] finish_client rounds={rounds} total_energy_j={total_energy:.3f}"
    )


if __name__ == "__main__":
    main()
