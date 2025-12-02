"""
Server-side orchestration for federated learning.

Coordinates all rounds:
- selecting clients
- downloading updates
- aggregating models
- computing final evaluation metrics
- generating summaries.
"""

import time
from src.fl.config.settings import settings
from src.fl.server.selection import select_all_clients, select_clients_aefl
from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl
)
from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import generate_cloud_summary
from src.fl.server.s3_io import (
    download_client_update,
    load_round_metadata,
    upload_global_model
)
from src.fl.server.utils_server import ROLES
from src.fl.models.gru_model import GRUModel
from src.fl.data.loader import load_test_loader_for_server
from src.fl.utils.logger import log_event


def main():
    dataset = settings.dataset
    rounds = settings.fl_rounds
    mode = settings.fl_mode.lower()

    print(f"[SERVER] Starting federated server | mode={mode.upper()}, dataset={dataset}")

    # Initialise global model
    model = GRUModel(hidden_size=settings.hidden_size)
    global_state = model.state_dict()
    upload_global_model(global_state, 1)

    # Evaluation loader
    test_loader = load_test_loader_for_server(dataset)

    # FL rounds
    for r in range(1, rounds + 1):
        print(f"\n========== ROUND {r} ==========")

        # Client selection
        if mode == "aefl" and r > 1:
            metadata = load_round_metadata(r - 1)
            chosen = select_clients_aefl(metadata, ROLES())
        else:
            chosen = select_all_clients(ROLES())

        print(f"[SERVER] Selected clients: {chosen}")

        # Collect updates from clients
        updates = {}
        start_wait = time.time()

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    upd = download_client_update(r, role)
                    if upd is not None:
                        updates[role] = upd
                        print(f"[SERVER] Received update from {role} ({len(updates)}/{len(chosen)})")

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > 300:
                print("[SERVER] WARNING: Timeout waiting for updates.")
                break

            time.sleep(1)

        # Aggregation
        start_aggr = time.time()

        if mode == "aefl":
            global_state = aggregate_aefl(updates)
        elif mode == "fedavg":
            global_state = aggregate_fedavg(updates)
        elif mode == "fedprox":
            global_state = aggregate_fedprox(updates)
        elif mode == "localonly":
            global_state = aggregate_fedavg(updates)
        else:
            raise ValueError(f"Unknown FL mode: {mode}")

        aggr_time = time.time() - start_aggr
        print(f"[SERVER] Aggregation complete | time={aggr_time:.3f}s")

        # Store next round model
        if r < rounds:
            upload_global_model(global_state, r + 1)

    # Final evaluation
    model.load_state_dict(global_state)
    metrics = evaluate_final_model(model, test_loader)

    print("\n[SERVER] Final Evaluation:")
    for k, v in metrics.items():
        print(f" {k} = {v:.6f}")

    generate_cloud_summary(metrics, rounds, mode)

    print(f"[SERVER] Training finished after {rounds} rounds.")


if __name__ == "__main__":
    main()
