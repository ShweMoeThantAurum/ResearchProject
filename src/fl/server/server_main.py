"""
Server-side orchestration for federated learning.

Coordinates:
- client selection (AEFL or all-clients)
- S3-based model distribution and update collection
- FedAvg/FedProx/AEFL aggregation
- final evaluation
- experiment summary generation.
"""

import time

from ..config.settings import settings
from .selection import select_all_clients, select_clients_aefl
from .aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)
from .evaluate import evaluate_final_model
from .summary import generate_cloud_summary
from .s3_io import (
    download_client_update,
    load_round_metadata,
    upload_global_model,
)
from .utils_server import ROLES
from ..models.gru_model import GRUModel
from ..utils.logger import log_event


def _aggregate(mode, updates):
    """Dispatch aggregation strategy based on FL mode."""
    mode = mode.lower()
    if mode == "aefl":
        return aggregate_aefl(updates)
    if mode == "fedavg":
        return aggregate_fedavg(updates)
    if mode == "fedprox":
        return aggregate_fedprox(updates)
    if mode == "localonly":
        return aggregate_fedavg(updates)
    raise ValueError("Unknown FL mode: " + mode)


def main():
    """Run the federated learning server for one full experiment."""
    dataset = settings.dataset
    rounds = settings.fl_rounds
    mode = settings.fl_mode.lower()

    print(
        f"[SERVER] Starting federated server | "
        f"mode={mode.upper()} dataset={dataset} rounds={rounds}"
    )
    log_event(
        f"[SERVER] start_server mode={mode} dataset={dataset} rounds={rounds}"
    )

    # Initialise global model on CPU
    model = GRUModel(hidden_size=settings.hidden_size)
    global_state = model.state_dict()

    # Upload first-round global model
    upload_global_model(global_state, round_id=1)

    # Orchestration loop
    for r in range(1, rounds + 1):
        print(f"\n========== ROUND {r} ==========")

        # Client selection (AEFL vs all-clients)
        if mode == "aefl" and r > 1:
            prev_meta = load_round_metadata(r - 1)
            chosen = select_clients_aefl(prev_meta)
        else:
            chosen = select_all_clients()

        print(f"[SERVER] Selected clients: {chosen}")
        log_event(
            f"[SERVER] round_select r={r} mode={mode} chosen={','.join(chosen)}"
        )

        # Collect updates from clients with a simple timeout
        updates = {}
        start_wait = time.time()
        timeout_s = 300

        print(f"[SERVER] Waiting for updates for round {r}...")
        while len(updates) < len(chosen):
            for role in chosen:
                if role in updates:
                    continue
                upd = download_client_update(r, role)
                if upd is not None:
                    updates[role] = upd
                    print(
                        f"[SERVER] Received update from {role} "
                        f"({len(updates)}/{len(chosen)})"
                    )

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout_s:
                print("[SERVER] WARNING: Timeout waiting for client updates.")
                break

            time.sleep(1.0)

        if not updates:
            print("[SERVER] ERROR: No client updates received; aborting.")
            log_event(f"[SERVER] no_updates r={r} abort_experiment=1")
            return

        # Aggregation
        start_aggr = time.time()
        global_state = _aggregate(mode, updates)
        aggr_time = time.time() - start_aggr

        print(f"[SERVER] Aggregation complete | time={aggr_time:.3f}s")
        log_event(
            f"[SERVER] aggregate_done r={r} mode={mode} time_s={aggr_time:.3f} "
            f"clients={len(updates)}"
        )

        # Upload next-round global model except after final round
        if r < rounds:
            upload_global_model(global_state, round_id=r + 1)

    # Final evaluation
    print("\n[SERVER] Running final evaluation...")
    final_metrics = evaluate_final_model(global_state, dataset)

    print("\n[SERVER] Final Evaluation:")
    for k, v in final_metrics.items():
        print(f" {k} = {v:.6f}")

    log_event(
        f"[SERVER] final_metrics dataset={dataset} mode={mode} "
        f"MAE={final_metrics.get('MAE', 0.0):.6f} "
        f"RMSE={final_metrics.get('RMSE', 0.0):.6f} "
        f"MAPE={final_metrics.get('MAPE', 0.0):.6f}"
    )

    # Save and upload experiment summary
    generate_cloud_summary(final_metrics, dataset, rounds, mode)

    print(f"[SERVER] Training finished after {rounds} rounds.")


if __name__ == "__main__":
    main()
