"""
Main FL engine — coordinates client selection, local training, aggregation, and energy logging.
"""

import os
import time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.core.client import client_indices, make_loader_for_client, train_local
from src.core.aggregation import (
    fedavg_average, adaptive_average, clone_state, set_state, pick_clients
)
from src.core.evaluator import init_csv, log_row, make_eval_loader, eval_model
from src.utils.config import load_config
from src.utils.seed import set_global_seed

from src.core.energy import cpu_mem_snapshot, state_size_bytes, compute_round_energy_j
from src.models.simple_gru import SimpleGRU

from src.utils.compression import (
    sparsify_state,
    topk_compress_state,
    quantize8_state,
)
from src.utils.privacy import dp_add_noise


# ==========================================================
# FedProx local training
# ==========================================================

def _train_local_fedprox(model, loader, device, epochs, lr, mu, global_state):
    model.train()
    loss_fn = torch.nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    global_params = {k: v.detach().clone().to(device) for k, v in global_state.items()}
    n_samples = 0

    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            yhat = model(x)
            loss = loss_fn(yhat, y)

            prox = 0.0
            for name, param in model.named_parameters():
                prox += torch.norm(param - global_params[name]) ** 2

            loss = loss + (mu / 2.0) * prox
            loss.backward()
            opt.step()
            n_samples += x.size(0)

    return {k: v.detach().clone() for k, v in model.state_dict().items()}, n_samples


# ==========================================================
# Model builder
# ==========================================================

def _build_model(model_type, num_nodes, hidden, in_timesteps=None, device="cpu"):
    if model_type.lower() != "simplegru":
        raise ValueError("This thesis build uses GRU only. Set model_type=SimpleGRU.")
    return SimpleGRU(num_nodes=num_nodes, hidden_size=hidden).to(device)


# ==========================================================
# Main function
# ==========================================================

def run_federated(config_path):
    cfg = load_config(config_path)
    set_global_seed(cfg.get("experiment", {}).get("seed", 42))

    exp = cfg["experiment"]
    data_cfg = cfg.get("data", {})
    fed = cfg.get("federation", {})
    train = cfg["training"]
    comp = cfg.get("compression", {})
    priv = cfg.get("privacy", {})
    eva = cfg["evaluation"]

    PROC = exp["proc_dir"]
    OUT = exp["output_dir"]
    os.makedirs(OUT, exist_ok=True)

    device = "mps" if torch.backends.mps.is_available() and eva.get("device_preference") == "mps" else "cpu"

    # Load meta.json
    meta_path = os.path.join(PROC, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Missing meta.json at {meta_path}. Preprocess first.")
    meta = json.load(open(meta_path))

    num_clients = meta["num_clients"]

    # Load eval datasets
    Xva = np.load(f"{PROC}/X_valid.npy")
    yva = np.load(f"{PROC}/y_valid.npy")
    Xte = np.load(f"{PROC}/X_test.npy")
    yte = np.load(f"{PROC}/y_test.npy")

    num_nodes = Xva.shape[-1]

    model = _build_model(
        exp["model_type"],
        num_nodes,
        train.get("hidden_size", 64),
        device=device
    )

    global_state = clone_state(model.state_dict())

    eval_valid = make_eval_loader(Xva, yva, batch=128)
    eval_test = make_eval_loader(Xte, yte, batch=128)

    rng = np.random.RandomState(exp.get("seed", 42))
    states = {
        str(i): {
            "energy": float(rng.uniform(0.4, 1.0)),
            "bandwidth": float(rng.uniform(0.4, 1.0))
        }
        for i in range(num_clients)
    }

    # CSV logging
    csv_path = os.path.join(OUT, "round_log.csv")
    init_csv(
        csv_path,
        ["round", "chosen", "val_mae", "val_rmse", "secs", "cpu_percent", "mem_mb",
         "energy_j", "bytes_sent_mb", "kept_ratio"]
    )

    rounds = fed.get("rounds", 20)
    local_epochs = fed.get("local_epochs", 1)
    max_part = fed.get("max_participants", num_clients)
    min_energy = fed.get("min_energy", 0.0)
    alpha = fed.get("alpha", 0.5)
    mu = fed.get("mu", 0.0)
    strategy = fed.get("strategy", "").lower()
    mode = exp.get("mode", "fedavg").lower()

    comp_enabled = comp.get("enabled", False)
    comp_strategy = (comp.get("strategy", "") or "").lower()
    sparsity = comp.get("sparsity", 0.0)
    k_frac = comp.get("k_frac", 0.1)

    dp_enabled = priv.get("enabled", False)
    dp_sigma = priv.get("dp_sigma", 0.0)

    total_energy_j, total_bytes_mb, chosen_sum = 0.0, 0.0, 0

    # ==========================================================
    # Training Loop
    # ==========================================================

    for r in range(1, rounds + 1):
        t0 = time.time()

        chosen = pick_clients(states, alpha, min_energy, max_part) if mode == "aefl" \
            else [str(i) for i in range(num_clients)]

        local_states, weights, scores = [], [], []
        bytes_per_client, comm_kept = [], []

        for cid in range(num_clients):
            cid_str = str(cid)
            if cid_str not in chosen:
                states[cid_str]["energy"] = min(1.0, states[cid_str]["energy"] + 0.02)
                continue

            idxs = client_indices(num_nodes, num_clients, cid)
            dl = make_loader_for_client(PROC, cid, idxs, num_nodes, batch=train.get("batch_size", 64))

            set_state(model, global_state)

            # Select training strategy
            if strategy == "fedprox":
                upd_state, n_i = _train_local_fedprox(
                    model, dl, device, local_epochs, train.get("lr", 1e-3),
                    mu, global_state
                )
            else:
                upd_state, n_i = train_local(
                    model, dl, device, epochs=local_epochs, lr=train.get("lr", 1e-3)
                )

            # Privacy
            if dp_enabled and dp_sigma > 0:
                upd_state = dp_add_noise(upd_state, dp_sigma)

            # Compression
            kept_ratio = 1.0
            comp_bytes = None

            if comp_enabled:
                if comp_strategy in ("magnitude", "prune", "sparsify"):
                    upd_state, kept_ratio, comp_bytes = sparsify_state(upd_state, sparsity)
                elif comp_strategy == "topk":
                    upd_state, kept_ratio, comp_bytes = topk_compress_state(upd_state, k_frac)
                elif comp_strategy in ("q8", "int8"):
                    upd_state, kept_ratio, comp_bytes = quantize8_state(upd_state)

            if comp_bytes is None:
                comp_bytes = state_size_bytes(upd_state)

            local_states.append(upd_state)
            weights.append(n_i)
            comm_kept.append(kept_ratio)
            bytes_per_client.append(comp_bytes)

            # Update energy score
            if mode == "aefl":
                states[cid_str]["energy"] = max(0.0, states[cid_str]["energy"] - 0.001 * kept_ratio)
                new_score = alpha * states[cid_str]["energy"] + (1 - alpha) * states[cid_str]["bandwidth"]
                scores.append(max(1e-6, new_score))

        # Aggregation
        if local_states:
            if mode == "aefl":
                global_state = adaptive_average(local_states, weights, scores)
            else:
                global_state = fedavg_average(local_states, weights)

        set_state(model, global_state)

        # Validation
        v_mae, v_rmse, *_ = eval_model(model, eval_valid, device)

        secs = time.time() - t0
        cpu_p, mem_mb = cpu_mem_snapshot()

        bytes_sent = sum(bytes_per_client)
        bytes_sent_mb = bytes_sent / (1024 * 1024)
        total_bytes_mb += bytes_sent_mb

        energy_j = compute_round_energy_j(secs, cpu_p, bytes_sent, "edge")
        total_energy_j += energy_j
        chosen_sum += len(chosen)

        print(f"Round {r:02d} | chosen={len(chosen)} | MAE={v_mae:.4f} | RMSE={v_rmse:.4f}")

        log_row(
            csv_path,
            [
                r, len(chosen), f"{v_mae:.6f}", f"{v_rmse:.6f}",
                f"{secs:.3f}", f"{cpu_p:.2f}", f"{mem_mb:.2f}",
                f"{energy_j:.6f}", f"{bytes_sent_mb:.6f}",
                f"{(np.mean(comm_kept) if comm_kept else 1.0):.4f}",
            ]
        )

    # ==========================================================
    # Final Test Evaluation
    # ==========================================================

    t_mae, t_rmse, *_ = eval_model(model, eval_test, device)

    results_path = os.path.join(OUT, "results.txt")
    ckpt_path = os.path.join(OUT, f"{exp['name'].lower().replace(' ', '_')}_state.pt")
    roundlog_path = csv_path

    with open(results_path, "w") as f:
        f.write(f"TEST MAE: {t_mae:.6f}\n")
        f.write(f"TEST RMSE: {t_rmse:.6f}\n")
        f.write(f"TOTAL ENERGY_J: {total_energy_j:.6f}\n")
        f.write(f"TOTAL BYTES_MB: {total_bytes_mb:.6f}\n")
        f.write(f"AVG CLIENTS PER ROUND: {chosen_sum / rounds:.3f}\n")

    torch.save(global_state, ckpt_path)

    print(f"Saved local outputs: {results_path}, {roundlog_path}, {ckpt_path}")

    # ==========================================================
    # UPLOAD TO S3
    # ==========================================================

    import boto3
    s3 = boto3.client("s3")
    bucket = "aefl-results"
    s3_prefix = f"experiments/{os.path.basename(OUT)}/"

    def upload(local, key):
        if os.path.exists(local):
            print(f"Uploading {local} → s3://{bucket}/{key}")
            s3.upload_file(local, bucket, key)
        else:
            print(f"WARNING: {local} not found, skipped.")

    upload(results_path, s3_prefix + "results.txt")
    upload(roundlog_path, s3_prefix + "round_log.csv")
    upload(ckpt_path, s3_prefix + "model_state.pt")

    print(f"S3 upload completed → s3://{bucket}/{s3_prefix}")
