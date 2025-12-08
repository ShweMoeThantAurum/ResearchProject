"""
Main server orchestration loop for adaptive federated learning.

Coordinates:
 - S3 cleanup,
 - global model initialisation,
 - federated training rounds,
 - aggregation (FedAvg / FedProx / AEFL),
 - final evaluation,
 - summary and energy aggregation.
"""

import os
import time
import torch

from src.fl.logger import log_event
from src.fl.utils import get_proc_dir, get_hidden_size, get_fl_rounds
from src.fl.server.init import (
    clear_round_data,
    infer_num_nodes,
    init_global_model,
    store_global_model,
)
from src.fl.server.s3 import load_client_update, load_round_metadata
from src.fl.server.selection import select_all_clients, select_clients_aefl
from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)
from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import generate_cloud_summary
from src.fl.server.modes import get_mode, is_aefl, is_fedavg, is_fedprox

ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]
PROC_DIR = get_proc_dir()


def main():
    """Run the main federated learning training loop on the server."""
    # Load experiment configuration from environment
    dataset = os.environ.get("DATASET", "unknown").lower()
    mode = get_mode()  # lowercase: "aefl", "fedavg", "fedprox"
    variant = os.environ.get("VARIANT_ID", "").strip()

    hidden_size = get_hidden_size()
    fl_rounds = get_fl_rounds()

    print(
        f"[SERVER] CONFIG | dataset={dataset}, mode={mode.upper()}, "
        f"variant={variant}, rounds={fl_rounds}, hidden={hidden_size}"
    )

    log_event(
        "server_config.log",
        {
            "dataset": dataset,
            "mode": mode,
            "variant": variant,
            "rounds": fl_rounds,
            "hidden_size": hidden_size,
        },
    )

    # --------------------------------------------------------
    # 1. Clean S3 directory for a fresh experiment
    # --------------------------------------------------------
    clear_round_data()

    # --------------------------------------------------------
    # 2. Initialise global model and upload for round 1
    # --------------------------------------------------------
    num_nodes = infer_num_nodes(PROC_DIR)
    global_state = init_global_model(num_nodes, hidden=hidden_size)
    store_global_model(global_state, round_id=1)

    # ========================================================
    # Federated Training Rounds
    # ========================================================
    for r in range(1, fl_rounds + 1):
        print(f"\n========== ROUND {r} ==========")

        # ===== CLIENT SELECTION =====
        if is_aefl(mode) and r > 1:
            # AEFL uses metadata from the previous round
            prev_meta = load_round_metadata(r - 1)
            chosen, scores = select_clients_aefl(prev_meta, ROLES, round_id=r)
        else:
            # Baselines (and AEFL round 1): all clients
            chosen = select_all_clients(ROLES)
            scores = {role: 1.0 for role in chosen}

        print(f"[SERVER] Selected clients: {chosen}")

        # ===== WAIT FOR CLIENT UPDATES =====
        updates = {}
        start_wait = time.time()
        timeout = 300  # seconds

        print(f"[SERVER] Waiting for updates for round {r}...")

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    upd = load_client_update(r, role)
                    if upd is not None:
                        updates[role] = upd
                        print(
                            f"[SERVER] Received {role} "
                            f"({len(updates)}/{len(chosen)})"
                        )

            if len(updates) == len(chosen):
                break
            if time.time() - start_wait > timeout:
                print("[SERVER] WARNING: Timeout waiting for updates.")
                break

            time.sleep(2)

        # ===== AGGREGATION =====
        start_aggr = time.time()

        if is_aefl(mode):
            global_state = aggregate_aefl(updates, scores)
            mode_label = "AEFL"
        elif is_fedavg(mode):
            global_state = aggregate_fedavg(updates)
            mode_label = "FedAvg"
        elif is_fedprox(mode):
            global_state = aggregate_fedprox(updates)
            mode_label = "FedProx"
        else:
            # Fallback to FedAvg if an unknown mode is supplied
            global_state = aggregate_fedavg(updates)
            mode_label = mode.upper()

        aggr_time = time.time() - start_aggr
        print(
            f"[SERVER] Aggregation complete | mode={mode_label}, "
            f"time={aggr_time:.3f}s"
        )

        log_event(
            "server_aggregation.log",
            {
                "round": r,
                "dataset": dataset,
                "mode": mode,
                "variant": variant,
                "aggregation_time_sec": aggr_time,
                "num_updates": len(updates),
            },
        )

        # ===== STORE GLOBAL FOR NEXT ROUND =====
        next_round = r + 1
        if next_round <= fl_rounds:
            store_global_model(global_state, next_round)

    # ========================================================
    # Final Evaluation
    # ========================================================
    metrics = evaluate_final_model(global_state, PROC_DIR, num_nodes, hidden_size)

    print("\n[SERVER] Final Evaluation (TEST SET):")
    for k, v in metrics.items():
        print(f" {k} = {v:.6f}")

    # Core CSV/JSON + energy_summary (variant-aware)
    generate_cloud_summary(metrics, fl_rounds, mode.upper())

    # Additional energy aggregation into outputs/<dataset>/<mode>/energy_summary.json
    try:
        from src.fl.server.energy import aggregate_energy_logs

        aggregate_energy_logs()
    except ImportError:
        # If energy aggregation is not available, silently skip
        pass

    print(f"[SERVER] Training finished after {fl_rounds} rounds.")


if __name__ == "__main__":
    main()
