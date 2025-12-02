"""
Main client loop for S3-based federated learning.

Each client:
    - Loads its local dataset partition
    - Repeatedly downloads the global model from S3
    - Trains locally for a few epochs
    - Applies optional DP and compression
    - Uploads the processed update and metadata
    - Logs detailed energy statistics
"""

import torch

from src.fl.utils.logger import log_event
from src.fl.models import SimpleGRU
from src.fl.client.utils_client import (
    get_client_role,
    get_fl_mode,
    client_allows_training,
    get_dataset_name,
    get_fl_rounds,
    get_batch_size,
    get_local_epochs,
    get_lr,
    get_hidden_size,
    get_energy_params,
    build_client_dataloader,
    infer_num_nodes_for_dataset,
)
from src.fl.client.train import train_one_round
from src.fl.client.energy import estimate_round_energy, approximate_flops_per_round
from src.fl.client.dp import maybe_add_dp_noise
from src.fl.client.compression import maybe_compress_state
from src.fl.client.comm import (
    download_global_model,
    upload_client_update,
    upload_client_metadata,
)


def build_metadata_record(role,
                          round_id,
                          energy_record,
                          train_loss,
                          train_samples,
                          update_bytes,
                          upload_latency_sec):
    """
    Build a compact metadata dictionary summarising this round.

    Includes:
        - training loss and samples
        - energy breakdown
        - communication volume
        - uplink bandwidth estimate
    """
    if upload_latency_sec <= 0.0:
        bandwidth_mbps = 0.0
    else:
        mbits = (update_bytes * 8.0) / 1e6
        bandwidth_mbps = mbits / upload_latency_sec

    return {
        "role": role,
        "round": round_id,
        "train_loss": float(train_loss),
        "train_samples": int(train_samples),
        "compute_time_j": float(energy_record.get("compute_time_j", 0.0)),
        "compute_flops_j": float(energy_record.get("compute_flops_j", 0.0)),
        "comm_j": float(energy_record.get("comm_j", 0.0)),
        "total_energy_j": float(energy_record.get("total_j", 0.0)),
        "download_mb": float(energy_record.get("download_mb", 0.0)),
        "upload_mb": float(energy_record.get("upload_mb", 0.0)),
        "bandwidth_mbps": bandwidth_mbps,
    }


def run_client():
    """
    Run the federated learning client for all configured rounds.

    This function is intended to be the entrypoint inside each
    Docker container launched by docker-compose.
    """
    role = get_client_role()
    mode = get_fl_mode()
    dataset_name = get_dataset_name()

    rounds = get_fl_rounds()
    batch_size = get_batch_size()
    local_epochs = get_local_epochs()
    lr = get_lr()
    hidden_size = get_hidden_size()

    device_power_watts, net_j_per_mb, flop_energy_j = get_energy_params()

    num_nodes = infer_num_nodes_for_dataset(dataset_name)
    seq_len = 12

    print("[%s] Starting client | mode=%s | dataset=%s | nodes=%d | rounds=%d"
          % (role, mode, dataset_name, num_nodes, rounds))

    loader = build_client_dataloader(
        dataset_name=dataset_name,
        role=role,
        batch_size=batch_size,
        num_nodes=num_nodes,
    )

    model = SimpleGRU(num_nodes=num_nodes,
                      hidden_size=hidden_size,
                      seq_len=seq_len)

    device = "cpu"
    total_energy_j = 0.0

    for r in range(1, rounds + 1):
        print("\n[%s] ===== ROUND %d =====" % (role, r))

        global_state, dl_bytes = download_global_model(
            round_id=r,
            role=role,
            dataset_name=dataset_name,
        )

        model.load_state_dict(global_state)

        if not client_allows_training(mode):
            updated_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
            train_time = 0.0
            train_loss = 0.0
            train_samples = 0
            train_flops = 0.0
        else:
            prox_ref_state = None
            if mode.lower() == "fedprox":
                prox_ref_state = {k: v.clone() for k, v in global_state.items()}

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

            train_flops = approximate_flops_per_round(
                total_samples=train_samples,
                num_nodes=num_nodes,
                hidden_size=hidden_size,
                seq_len=seq_len,
            )

        dp_state = maybe_add_dp_noise(updated_state)
        comp_state, kept_ratio, modeled_bytes = maybe_compress_state(dp_state)

        up_bytes, up_latency = upload_client_update(
            round_id=r,
            role=role,
            state_dict=comp_state,
            dataset_name=dataset_name,
        )

        energy_record = estimate_round_energy(
            role=role,
            round_id=r,
            train_time_sec=train_time,
            train_flops=train_flops,
            download_bytes=dl_bytes,
            upload_bytes=up_bytes,
            device_power_watts=device_power_watts,
            net_j_per_mb=net_j_per_mb,
            flop_energy_j=flop_energy_j,
        )

        total_energy_j = total_energy_j + energy_record["total_j"]

        meta = build_metadata_record(
            role=role,
            round_id=r,
            energy_record=energy_record,
            train_loss=train_loss,
            train_samples=train_samples,
            update_bytes=up_bytes,
            upload_latency_sec=up_latency,
        )

        upload_client_metadata(
            round_id=r,
            role=role,
            meta=meta,
            dataset_name=dataset_name,
        )

    log_event("client_energy_summary.log", {
        "role": role,
        "rounds": rounds,
        "total_energy_j": total_energy_j,
        "mode": mode,
        "dataset": dataset_name,
    })

    print("[%s] Finished %d rounds. Total estimated energy=%.2f J."
          % (role, rounds, total_energy_j))


if __name__ == "__main__":
    run_client()
