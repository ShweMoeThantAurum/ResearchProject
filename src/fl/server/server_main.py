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
from src.fl.config.config_loader import load_experiment_config
from src.fl.server.selection import select_all_clients, select_clients_aefl
from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)
from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import (
    generate_cloud_summary,
    log_round_summary,
)
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


def _collect_final_energy(dataset, rounds, roles):
    """
    Collect total energy usage for each role from final-round metadata.
    """
    meta = load_round_metadata(rounds)

    totals = {}
    for role in roles:
        r_meta = meta.get(role, {})
        totals[role] = float(r_meta.get("total_energy_j", 0.0))

    print("\n[SERVER] Final Energy Totals (J):")
    for role, value in totals.items():
        print(f"  {role:10s}: {value:.3f}")

    return totals


def main():
    """Run the federated learning server loop."""
    # --------------------------------------------
    # Load YAML configuration (settings + overlays)
    # --------------------------------------------
    load_experiment_config()

    dataset = get_dataset()
    mode = get_fl_mode()
    rounds = settings.fl_rounds
    roles = ["roadside", "vehicle", "sensor", "camera", "bus"]

    clear_all_rounds()
    num_nodes = _infer_num_nodes(dataset)

    print(
        f"[SERVER] Starting federated server | "
        f"mode={mode.upper()} dataset={dataset} rounds={rounds} nodes={num_nodes}"
    )

    # --------------------------------------------
    # Init global model
    # --------------------------------------------
    model = GRUModel(num_nodes=num_nodes, hidden_size=settings.hidden_size)
    global_state = model.state_dict()
    upload_global_model(1, global_state)

    # --------------------------------------------
    # Training rounds
    # --------------------------------------------
    for r in range(1, rounds + 1):
        print(f"\n========== ROUND {r} ==========")

        # ----- Client Selection -----
        if mode == "aefl" and r > 1:
            metadata = load_round_metadata(r - 1)
            chosen = select_clients_aefl(metadata)
        else:
            chosen = select_all_clients()

        print(f"[SERVER] Selected clients: {chosen}")

        # ----- Collect Updates -----
        updates = {}
        start_wait = time.time()
        timeout = 300

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    upd = download_client_update(r, role)
                    if upd is not None:
                        updates[role] = upd
                        print(f"[SERVER] Received update from {role} ({len(updates)}/{len(chosen)})")

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout:
                print("[SERVER] WARNING: Timeout waiting for client updates.")
                break

            time.sleep(1)

        # ----- Aggregation -----
        start_aggr = time.time()

        if not updates:
            print("[SERVER] WARNING: no updates received; reusing previous global state.")
        else:
            if mode == "aefl":
                global_state = aggregate_aefl(updates)
            elif mode == "fedavg":
                global_state = aggregate_fedavg(updates)
            elif mode == "fedprox":
                global_state = aggregate_fedprox(updates)
            elif mode == "localonly":
                global_state = aggregate_fedavg(updates)
            else:
                raise ValueError(f"Unknown mode: {mode}")

        aggr_time = time.time() - start_aggr
        print(f"[SERVER] Aggregation complete | time={aggr_time:.6f}s")

        # Log per-round summary
        log_round_summary(
            round_id=r,
            selected_clients=chosen,
            num_updates=len(updates),
            aggregation_time_s=aggr_time,
            mode_label=mode,
        )

        # Upload global model
        if r < rounds:
            upload_global_model(r + 1, global_state)

    # --------------------------------------------
    # Final evaluation
    # --------------------------------------------
    metrics = evaluate_final_model(global_state, dataset)

    print("\n[SERVER] Final Evaluation:")
    for k, v in metrics.items():
        print(f" {k} = {v:.6f}")

    # --------------------------------------------
    # Final energy collection from metadata
    # --------------------------------------------
    energy_totals = _collect_final_energy(dataset, rounds, roles)

    # --------------------------------------------
    # Save summary + metrics + energy
    # --------------------------------------------
    generate_cloud_summary(
        final_metrics=metrics,
        rounds=rounds,
        mode_label=mode,
        energy_totals=energy_totals,
    )

    print(f"[SERVER] Training finished after {rounds} rounds.")


if __name__ == "__main__":
    main()
