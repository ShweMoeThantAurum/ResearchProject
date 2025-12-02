"""
Client-side entrypoint for federated learning.

Each client loads its dataset partition, receives the global model,
performs local training, computes energy usage, applies optional
compression or DP noise, and uploads its update + metadata to S3.
"""

import time
from src.fl.config.settings import settings
from src.fl.data.loader import load_client_partition
from src.fl.models.gru_model import GRUModel
from src.fl.client.train import train_one_round
from src.fl.client.energy import compute_energy
from src.fl.client.comm import (
    download_global_model,
    upload_update,
    upload_metadata
)
from src.fl.client.dp import apply_dp_noise
from src.fl.client.compression import compress_update
from src.fl.utils.logger import log_event


def main():
    role = settings.client_role
    dataset = settings.dataset
    rounds = settings.fl_rounds

    log_event(role, f"Starting client for dataset={dataset}")

    # Load local dataset partition
    loader = load_client_partition(dataset, role)

    # Initialise model
    model = GRUModel(hidden_size=settings.hidden_size)

    total_energy = 0.0

    for r in range(1, rounds + 1):
        print(f"[{role}] ===== ROUND {r} =====")

        # Download global model parameters
        state = download_global_model(r)
        model.load_state_dict(state)

        # Train locally for one round
        start = time.time()
        loss = train_one_round(model, loader, settings)
        train_time = time.time() - start

        # Create update dictionary
        update = {k: v.cpu() for k, v in model.state_dict().items()}

        # Optional DP
        if settings.dp_enabled:
            update = apply_dp_noise(update, sigma=settings.dp_sigma)

        # Optional Compression
        if settings.compression_enabled:
            update = compress_update(update, settings)

        # Compute energy
        energy, breakdown = compute_energy(
            compute_time=train_time,
            model_size_mb=settings.model_size_mb,
            batch_energy=settings.device_power_watts,
            net_energy=settings.net_j_per_mb
        )
        total_energy += energy

        # Upload update + metadata
        upload_update(role, r, update)
        upload_metadata(role, r, {
            "bandwidth_mbps": breakdown["bandwidth"],
            "total_energy_j": energy
        })

        print(
            f"[{role}] Round {r} | loss={loss:.6f}, "
            f"time={train_time:.3f}s, energy={energy:.3f} J"
        )

    print(f"[{role}] Finished {rounds} rounds. Total estimated energy={total_energy:.2f} J.")


if __name__ == "__main__":
    main()
