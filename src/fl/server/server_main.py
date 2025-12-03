"""
Server-side orchestration for federated learning.

Coordinates all training rounds:
- selecting clients
- downloading updates
- aggregating models
- computing final evaluation metrics
- generating summaries
"""

import os
import time
import torch

from src.fl.config.settings import settings
from src.fl.server.selection import select_all_clients, select_clients_aefl
from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)
from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import generate_cloud_summary
from src.fl.server.s3_io import (
    download_client_update,
    load_round_metadata,
    upload_global_model,
    clear_all_rounds,
)
from src.fl.server.utils_server import get_dataset, get_fl_mode
from src.fl.models.gru_model import GRUModel


def _infer_num_nodes(dataset: str) -> int:
    """Infer number of nodes from processed train split."""
    path = os.path.join("datasets", "processed", dataset, "global", "train.pt")
    X_train, _ = torch.load(path)
    return X_train.shape[2]


def main():
    """Run the federated learning server loop."""
    dataset = get_dataset()
    mode = get_fl_mode()
    rounds = settings.fl_rounds

    # Clean previous S3 objects for this experiment to avoid stale globals
    clear_all_rounds()

    num_nodes = _infer_num_nodes(dataset)

    print(
        f"[SERVER] Starting federated server | "
        f"mode={mode.upper()} dataset={dataset} rounds={rounds} nodes={num_nodes}"
    )

    # Initialise global model and upload for round 1
    model = GRUModel(num_nodes=num_nodes, hidden_size=settings.hidden_size)
    global_state = model.state_dict()
    upload_global_model(1, global_state)

    # Training rounds
    for r in range(1, rounds + 1):
        print(f"\n========== ROUND {r} ==========")

        # ------------------------------------------------
        # Client selection
        # ------------------------------------------------
        if mode == "aefl" and r > 1:
            metadata = load_round_metadata(r - 1)
            chosen = select_clients_aefl(metadata)
        else:
            chosen = select_all_clients()

        print(f"[SERVER] Selected clients: {chosen}")

        # ------------------------------------------------
        # Collect client updates
        # ------------------------------------------------
        updates = {}
        start_wait = time.time()
        timeout = 300  # seconds

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    upd = download_client_update(r, role)
                    if upd is not None:
                        updates[role] = upd
                        print(
                            f"[SERVER] Received update from {role} "
                            f"({len(updates)}/{len(chosen)})"
                        )

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout:
                print("[SERVER] WARNING: Timeout waiting for updates.")
                break

            time.sleep(1.0)

        # ------------------------------------------------
        # Aggregation
        # ------------------------------------------------
        start_aggr = time.time()

        if not updates:
            # No updates received → reuse previous global state
            print(
                "[SERVER] WARNING: No client updates received this round; "
                "reusing previous global model."
            )
            # global_state stays as previous
        else:
            if mode == "aefl":
                global_state = aggregate_aefl(updates)
            elif mode == "fedavg":
                global_state = aggregate_fedavg(updates)
            elif mode == "fedprox":
                global_state = aggregate_fedprox(updates)
            elif mode == "localonly":
                # still aggregate, but conceptually represents local-only baseline
                global_state = aggregate_fedavg(updates)
            else:
                raise ValueError(f"Unknown FL mode: {mode}")

        aggr_time = time.time() - start_aggr
        print(f"[SERVER] Aggregation complete | time={aggr_time:.3f}s")

        # Upload next-round global model
        next_round = r + 1
        if next_round <= rounds:
            upload_global_model(next_round, global_state)

    # ------------------------------------------------
    # Final evaluation on test set
    # ------------------------------------------------
    metrics = evaluate_final_model(global_state, dataset)

    print("\n[SERVER] Final Evaluation:")
    for k, v in metrics.items():
        print(f" {k} = {v:.6f}")

    generate_cloud_summary(metrics, rounds, mode)

    print(f"[SERVER] Training finished after {rounds} rounds.")


if __name__ == "__main__":
    main()
