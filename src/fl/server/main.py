"""Main cloud FL server coordinating AEFL, FedAvg, FedProx, and LocalOnly."""

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
from src.fl.server.modes import get_mode, is_aefl, is_fedavg, is_fedprox, is_localonly


ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]

PROC_DIR = get_proc_dir()
HIDDEN_SIZE = get_hidden_size()
FL_MODE = get_mode()
FL_ROUNDS = get_fl_rounds()

print(f"[SERVER] CONFIG: mode={FL_MODE.upper()}, rounds={FL_ROUNDS}, hidden={HIDDEN_SIZE}")


def main():
    """
    Main cloud-based federated learning orchestration loop.

    This function:
      - cleans S3
      - initialises the global model
      - runs FL rounds
      - performs aggregation
      - evaluates the final model
      - generates summary outputs
    """
    print(f"[SERVER] Starting server | mode={FL_MODE.upper()}")

    # Cleanup S3
    clear_round_data()

    # Init model
    num_nodes = infer_num_nodes(PROC_DIR)
    global_state = init_global_model(num_nodes, hidden=HIDDEN_SIZE)

    # Upload initial model for Round 1
    store_global_model(global_state, round_id=1)

    # ------------------------------------------------------------
    # Federated Learning Rounds
    # ------------------------------------------------------------
    for r in range(1, FL_ROUNDS + 1):
        print(f"\n========== ROUND {r} ==========")

        # AEFL adaptive client selection (from previous round metadata)
        if is_aefl(FL_MODE) and r > 1:
            prev_meta = load_round_metadata(r - 1)
            chosen = select_clients_aefl(prev_meta, ROLES)
            print(f"[SERVER] AEFL selected: {chosen}")
        else:
            chosen = select_all_clients(ROLES)
            print(f"[SERVER] All clients chosen: {chosen}")

        # --------------------------------------------------------
        # Wait for client updates
        # --------------------------------------------------------
        updates = {}
        start_wait = time.time()
        timeout = 300

        print(f"[SERVER] Waiting for updates for round {r}...")

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    upd = load_client_update(r, role)
                    if upd is not None:
                        updates[role] = upd
                        print(f"[SERVER] Received {role} ({len(updates)}/{len(chosen)})")

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout:
                print("[SERVER] WARNING: Timeout waiting for updates.")
                break

            time.sleep(2)

        # --------------------------------------------------------
        # Aggregation
        # --------------------------------------------------------
        start_aggr = time.time()

        if is_aefl(FL_MODE):
            global_state = aggregate_aefl(updates)
            mode_label = "AEFL"

        elif is_fedavg(FL_MODE):
            global_state = aggregate_fedavg(updates)
            mode_label = "FedAvg"

        elif is_fedprox(FL_MODE):
            global_state = aggregate_fedprox(updates)
            mode_label = "FedProx"

        elif is_localonly(FL_MODE):
            # "LocalOnly" still uses FedAvg aggregation
            global_state = aggregate_fedavg(updates)
            mode_label = "LocalOnly"

        else:
            global_state = aggregate_fedavg(updates)
            mode_label = FL_MODE

        aggr_time = time.time() - start_aggr
        print(f"[SERVER] Aggregation complete | mode={mode_label}, time={aggr_time:.3f}s")

        # --------------------------------------------------------
        # Upload NEW aggregated model for the NEXT round
        # --------------------------------------------------------
        next_round = r + 1
        if next_round <= FL_ROUNDS:
            store_global_model(global_state, next_round)

    # ------------------------------------------------------------
    # Final Evaluation
    # ------------------------------------------------------------
    metrics = evaluate_final_model(global_state, PROC_DIR, num_nodes, HIDDEN_SIZE)

    print("\n[SERVER] Final Evaluation:")
    for k, v in metrics.items():
        print(f" {k} = {v:.6f}")

    generate_cloud_summary(metrics, FL_ROUNDS, FL_MODE.upper())

    print(f"[SERVER] Training finished after {FL_ROUNDS} rounds.")


if __name__ == "__main__":
    main()
