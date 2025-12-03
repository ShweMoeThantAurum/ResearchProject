"""
Federated Learning Server.
Coordinates dataset-wide FL training:
- Uploads initial model
- Selects clients each round
- Collects updates
- Aggregates
- Writes summaries
- Performs final evaluation
"""

import time
from src.fl.config.settings import settings
from src.fl.server.selection import select_clients
from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)
from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import save_experiment_summary
from src.fl.server.s3_io import (
    upload_global_model,
    download_client_update,
    load_round_metadata,
)
from src.fl.models.gru_model import GRUModel
from src.fl.data.loader import load_test_loader_for_server
from src.fl.utils.logger import log_event
from src.fl.server.utils_server import ROLES


def main():
    dataset = settings.dataset
    mode = settings.fl_mode
    rounds = settings.fl_rounds

    print(f"[SERVER] Starting server | dataset={dataset} mode={mode.upper()} rounds={rounds}")

    # Build initial GRU model
    # Load from test loader to detect num_nodes
    test_loader = load_test_loader_for_server(dataset)
    first_batch = next(iter(test_loader))
    X0, _ = first_batch
    num_nodes = X0.size(2)

    model = GRUModel(num_nodes=num_nodes, hidden_size=settings.hidden_size)
    global_state = model.state_dict()

    # Upload round 1 model
    upload_global_model(1, global_state)

    # FL rounds
    for r in range(1, rounds + 1):
        print(f"\n========== ROUND {r} ==========")

        # AEFL / FedAvg / FedProx / LocalOnly
        if r == 1:
            selected = list(ROLES)
        else:
            metadata = load_round_metadata(r - 1)
            selected = select_clients(metadata, mode)

        print("[SERVER] Selected clients:", selected)

        # Wait for updates
        updates = {}
        wait_start = time.time()

        while len(updates) < len(selected):
            for role in selected:
                if role not in updates:
                    upd = download_client_update(r, role)
                    if upd is not None:
                        updates[role] = upd
                        print(f"[SERVER] Received {role} ({len(updates)}/{len(selected)})")

            if len(updates) == len(selected):
                break

            if time.time() - wait_start > 300:
                print("[SERVER] Timeout collecting updates.")
                break

            time.sleep(1)

        # Aggregation
        if mode == "aefl":
            global_state = aggregate_aefl(updates)
        elif mode == "fedavg":
            global_state = aggregate_fedavg(updates)
        elif mode == "fedprox":
            global_state = aggregate_fedprox(updates)
        elif mode == "localonly":
            global_state = aggregate_fedavg(updates)
        else:
            raise ValueError("Unknown FL mode:", mode)

        print("[SERVER] Aggregation complete")

        # Upload next-round model
        if r < rounds:
            upload_global_model(r + 1, global_state)

    # Final evaluation
    model.load_state_dict(global_state)
    metrics = evaluate_final_model(model, dataset)

    print("\n[SERVER] Final evaluation:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.6f}")

    save_experiment_summary(dataset, mode, metrics)

    print(f"[SERVER] Training finished.")
