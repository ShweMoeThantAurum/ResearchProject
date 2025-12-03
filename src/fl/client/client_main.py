"""
Client entrypoint for federated learning.

Each client:
- loads its local data split
- downloads the global model from S3
- trains locally
- applies optional DP and compression
- uploads its update and energy/bandwidth metadata.
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
from src.fl.client.utils_client import compute_bandwidth_mbps
from src.fl.utils.logger import log_event


def main():
    """Run the FL client loop for all rounds."""
    role = settings.client_role or "roadside"
    dataset = settings.dataset
    mode = settings.fl_mode
    rounds = settings.fl_rounds

    log_event(f"[{role}] Starting client | dataset={dataset} mode={mode}")
    print(f"[{role}] Starting client | dataset={dataset} mode={mode} rounds={rounds}")

    # For this thesis implementation we assume CPU.
    # GPU-aware extensions are left as future work.
    device = torch.device("cpu")

    # Build local DataLoader for this client partition
    loader = load_client_partition(dataset, role, settings.batch_size)

    total_energy = 0.0

    for r in range(1, rounds + 1):
        print(f"[{role}] ===== ROUND {r} =====")

        # Download global model state for this round
        state_dict = download_global_model(r, role)

        # Infer number of nodes from decoder layer
        decoder_weight = state_dict["decoder.weight"]
        hidden_size = decoder_weight.shape[1]
        num_nodes = decoder_weight.shape[0]

        model = GRUModel(num_nodes=num_nodes, hidden_size=hidden_size)
        model.load_state_dict(state_dict)

        # Local training
        start_time = time.time()
        updated_state, loss, num_samples, approx_flops = train_one_round(
            model=model,
            loader=loader,
            role=role,
            round_id=r,
            device=device,
            local_epochs=settings.local_epochs,
            lr=settings.lr,
            mode=mode,
            global_state=state_dict if mode == "fedprox" else None,
        )
        train_time = time.time() - start_time

        # DP noise (client-side privacy)
        if settings.dp_enabled:
            updated_state = apply_dp_noise(updated_state, sigma=settings.dp_sigma)

        # Compression (optional communication saving)
        if settings.compression_enabled:
            updated_state = compress_update(updated_state, settings)

        # Upload update to S3
        update_bytes, upload_latency = upload_update(r, role, updated_state)

        # Energy accounting for this round
        e_comp, e_comm, e_total = compute_energy(
            compute_duration_s=train_time,
            update_size_bytes=update_bytes,
            device_power_watts=settings.device_power_watts,
            net_j_per_mb=settings.net_j_per_mb,
        )
        total_energy += e_total

        # Simple bandwidth estimate based on upload
        bandwidth_mbps = compute_bandwidth_mbps(update_bytes, upload_latency)

        # Minimal metadata required for AEFL selection
        meta = {
            "bandwidth_mbps": bandwidth_mbps,
            "total_energy_j": e_total,
            "train_loss": float(loss),
            "samples": int(num_samples),
            "approx_flops": float(approx_flops),
        }

        upload_metadata(r, role, meta)

        print(
            f"[{role}] Round {r} | loss={loss:.6f}, "
            f"time={train_time:.3f}s, energy_total={e_total:.3f} J "
            f"(compute={e_comp:.3f} J, comm={e_comm:.3f} J)"
        )

        log_event(
            f"[{role}] round={r} "
            f"loss={loss:.6f} time={train_time:.3f}s "
            f"E_comp={e_comp:.4f}J E_comm={e_comm:.4f}J E_total={e_total:.4f}J "
            f"BW={bandwidth_mbps:.3f}Mb/s samples={num_samples}"
        )

    print(f"[{role}] Finished {rounds} rounds. Total estimated energy={total_energy:.2f} J.")
    log_event(f"[{role}] finished rounds={rounds} total_energy_j={total_energy:.4f}")
