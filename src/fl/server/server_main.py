"""
Main server orchestration loop for federated learning.
Coordinates S3 cleanup, per-round client updates, aggregation, and evaluation.
"""

import time

from ..models.gru_model import GRUModel
from ..utils.logger import log_event
from .utils_server import (
    get_dataset,
    get_fl_mode,
    get_fl_rounds,
    get_hidden_size,
    ROLES,
)
from .aggregation import aggregate_fedavg, aggregate_fedprox, aggregate_aefl
from .selection import select_all_clients, select_clients_aefl
from .s3_io import clear_all_rounds, upload_global_model, download_client_update, load_round_metadata
from .evaluate import evaluate_global_model
from .summary import save_experiment_summary


def _init_global_state(num_nodes, hidden_size):
    """Initialise GRU model and return its state dict."""
    model = GRUModel(num_nodes=num_nodes, hidden_size=hidden_size)
    return model.state_dict()


def _infer_num_nodes_from_dataset(dataset):
    """Infer node count by loading a test loader."""
    from ..data.loader import load_test_loader_for_server
    loader, num_nodes = load_test_loader_for_server(dataset, batch_size=1)
    return num_nodes


def main():
    """Run the FL server training loop."""
    dataset = get_dataset()
    mode = get_fl_mode()
    rounds = get_fl_rounds()
    hidden_size = get_hidden_size()

    print(f"[SERVER] Starting | dataset={dataset}, mode={mode}, rounds={rounds}, hidden={hidden_size}")
    log_event(f"[SERVER] Starting | dataset={dataset}, mode={mode}, rounds={rounds}, hidden={hidden_size}")

    clear_all_rounds()

    num_nodes = _infer_num_nodes_from_dataset(dataset)
    print(f"[SERVER] Inferred num_nodes={num_nodes}")

    global_state = _init_global_state(num_nodes, hidden_size)

    upload_global_model(round_id=1, state_dict=global_state)

    round_records = []

    for r in range(1, rounds + 1):
        print(f"\n[SERVER] ===== ROUND {r} =====")

        if mode == "AEFL" and r > 1:
            meta_prev = load_round_metadata(r - 1)
            chosen = select_clients_aefl(meta_prev)
        else:
            chosen = select_all_clients()

        print(f"[SERVER] Selected clients: {chosen}")

        updates = {}
        start_wait = time.time()
        timeout_sec = 300

        # Wait for all selected client updates
        while len(updates) < len(chosen):
            for role in chosen:
                if role in updates:
                    continue
                state = download_client_update(r, role)
                if state is not None:
                    updates[role] = state
                    print(f"[SERVER] Received update from {role} ({len(updates)}/{len(chosen)})")

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout_sec:
                print("[SERVER] WARNING: Timeout waiting for updates.")
                break

            time.sleep(2)

        if not updates:
            print("[SERVER] No updates received this round; stopping early.")
            break

        # Aggregate according to mode
        if mode == "FedAvg":
            global_state = aggregate_fedavg(updates)
        elif mode == "FedProx":
            global_state = aggregate_fedprox(updates)
        elif mode == "AEFL":
            global_state = aggregate_aefl(updates)
        elif mode == "LocalOnly":
            global_state = aggregate_fedavg(updates)
        else:
            global_state = aggregate_fedavg(updates)

        # Upload next global model if more rounds remain
        if r < rounds:
            upload_global_model(round_id=r + 1, state_dict=global_state)

        round_records.append({
            "round": r,
            "num_selected": len(chosen),
            "selected_clients": ",".join(chosen),
            "num_updates": len(updates),
        })

    # Final evaluation
    final_metrics = evaluate_global_model(global_state, dataset)

    print("\n[SERVER] Final evaluation metrics:")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.6f}")

    save_experiment_summary(final_metrics, round_records)
    print(f"[SERVER] Training finished after {len(round_records)} effective rounds.")


if __name__ == "__main__":
    main()
