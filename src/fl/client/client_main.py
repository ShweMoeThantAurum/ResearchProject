"""
Client-side federated learning loop.

Each client:
    - Waits for the global model
    - Trains locally for one round
    - Applies optional DP and compression
    - Computes energy consumption
    - Uploads the update + metadata to S3
"""

import os
import time
import json
import torch
import numpy as np

from src.fl.client.train import run_local_training
from src.fl.client.energy import compute_energy
from src.fl.client.comm import download_global_model, upload_update
from src.fl.client.dp import apply_dp_noise
from src.fl.client.compression import apply_compression
from src.fl.client.utils_client import load_client_dataset, print_flush
from src.fl.utils.serialization import load_torch
from src.fl.server.s3_io import upload_json


def main():
    role = os.environ.get("CLIENT_ROLE")
    dataset = os.environ.get("DATASET")
    mode = os.environ.get("FL_MODE", "AEFL")
    bucket = os.environ.get("S3_BUCKET", "aefl")

    rounds = int(os.environ.get("FL_ROUNDS", 20))
    hidden = int(os.environ.get("HIDDEN_SIZE", 64))

    proc_dir = os.path.join("datasets", "processed", dataset)

    print_flush("[%s] Starting client | mode=%s rounds=%d" % (role, mode, rounds))

    # Load local dataset partition
    X_train, y_train = load_client_dataset(proc_dir, role)

    # Tracking accumulated energy
    total_energy = 0.0

    for r in range(1, rounds + 1):
        print_flush("\n[%s] ===== ROUND %d =====" % (role, r))

        # 1. Wait for global model
        print_flush("[%s] Waiting for global model round %d..." % (role, r))
        global_state = download_global_model(bucket, dataset, mode, r)

        # 2. Local training
        update, train_loss, train_time = run_local_training(
            global_state,
            X_train,
            y_train,
            hidden
        )

        print_flush("[%s] Round %d training | loss=%.6f time=%.3fs samples=%d"
                    % (role, r, train_loss, train_time, len(X_train)))

        # 3. Energy computation
        energy = compute_energy(train_time, update)
        total_energy += energy["total_energy_j"]

        print_flush("[%s] Energy round %d: compute_time=%.2f J, compute_flops=%.2f J, comm_total=%.4f J, total=%.4f J"
                    % (role, r, energy["compute_time_j"], energy["compute_flops_j"],
                       energy["comm_energy_j"], energy["total_energy_j"]))

        # 4. DP noise
        if os.environ.get("DP_ENABLED", "false").lower() == "true":
            update = apply_dp_noise(update)

        # 5. Compression
        if os.environ.get("COMPRESSION_ENABLED", "false").lower() == "true":
            update = apply_compression(update)

        # 6. Upload update
        upload_update(bucket, dataset, mode, r, role, update)

        # 7. Upload metadata
        meta = {
            "bandwidth_mbps": energy["bandwidth_mbps"],
            "total_energy_j": energy["total_energy_j"]
        }
        upload_json(bucket, dataset, mode, r, "%s_metadata.json" % role, meta)

        print_flush("[%s] Uploaded metadata for round %d: bandwidth=%.3f Mb/s total_energy=%.2f J"
                    % (role, r, meta["bandwidth_mbps"], meta["total_energy_j"]))

    print_flush("[%s] Finished %d rounds. Total estimated energy=%.2f J."
                % (role, rounds, total_energy))


if __name__ == "__main__":
    main()
