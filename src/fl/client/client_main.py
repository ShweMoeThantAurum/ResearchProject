"""
Client entrypoint for local training and update upload.
Handles:
- model loading
- receiving global model
- running local training
- applying DP or compression
- energy accounting
- uploading updates and metadata
"""

import os
import json
import base64

import torch
import numpy as np

from src.fl.config import settings
from src.fl.models import SimpleGRU
from src.fl.client.train import train_one_round
from src.fl.client.energy import compute_energy_round
from src.fl.client.comm import load_global_state, upload_update, upload_metadata
from src.fl.client.dp import apply_dp
from src.fl.client.compression import apply_compression
from src.fl.client.utils_client import load_client_data, decode_state_dict


def main():
    role = os.environ.get("CLIENT_ROLE", "unknown")
    dataset = settings.get_dataset()
    rounds = settings.get_fl_rounds()

    print("[{}] Starting client | dataset={} rounds={}".format(role, dataset, rounds))

    seq_x, seq_y = load_client_data(dataset, role)

    # Local model
    model = SimpleGRU(
        input_dim=1,
        hidden_size=settings.get_hidden_size(),
        num_layers=1
    )

    total_energy = 0.0

    # Training loop
    for r in range(1, rounds + 1):
        print("[{}] ===== ROUND {} =====".format(role, r))

        # Receive global model
        global_state = load_global_state(r)
        if global_state is not None:
            model.load_state_dict(decode_state_dict(global_state))

        # Train locally
        loss, train_time, flops = train_one_round(model, seq_x, seq_y)

        # Create update dict
        update = {k: v.cpu() for k, v in model.state_dict().items()}

        # Differential privacy
        if settings.dp_enabled():
            update = apply_dp(update)

        # Compression
        if settings.compression_enabled():
            update = apply_compression(update)

        # Upload update
        upload_update(update, role, r)

        # Compute communication metadata
        comm_size_mb = np.sum([p.numel() * 4 for p in update.values()]) / (1024 * 1024)

        energy = compute_energy_round(
            train_time=train_time,
            flops_j=flops,
            comm_mb=comm_size_mb
        )

        total_energy += energy

        # Upload metadata
        upload_metadata(
            role=role,
            round_id=r,
            bandwidth_mbps=comm_size_mb / max(train_time, 1e-3),
            total_energy=energy
        )

        print("[{}] Round {} training | loss={:.6f} time={:.3f}s samples={}".format(
            role, r, loss, train_time, len(seq_x)
        ))
        print("[{}] Energy round {}: {:.4f} J".format(role, r, energy))

    print("[{}] Finished {} rounds. Total estimated energy={:.2f} J.".format(
        role, rounds, total_energy
    ))
