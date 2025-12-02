"""
Server-side orchestration for federated learning.

Responsibilities:
    - Clean previous round data
    - Initialise global model
    - Loop through FL rounds:
        • client selection
        • client update collection (S3 polling)
        • aggregation
        • saving next-round model
    - Final evaluation
    - Save experiment summary
"""

import os
import time
import torch

from src.fl.server.utils_server import (
    clear_round_data,
    infer_num_nodes,
    get_mode,
    get_fl_rounds,
    get_hidden_size,
    get_processed_dir,
)

from src.fl.server.selection import (
    select_all_clients,
    select_clients_aefl,
)

from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)

from src.fl.server.s3_io import (
    load_client_update,
    load_round_metadata,
    store_global_model,
)

from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import generate_cloud_summary

from src.fl.models import SimpleGRU


ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def main():
    mode = get_mode()
    rounds = get_fl_rounds()
    hidden = get_hidden_size()

    dataset = os.environ.get("DATASET")
    bucket = os.environ.get("S3_BUCKET", "aefl")

    print("[SERVER] Starting | mode=%s rounds=%d dataset=%s" %
          (mode.upper(), rounds, dataset))

    # Clean old data
    clear_round_data(dataset, mode)

    # Load dataset shape (how many nodes?)
    proc_dir = get_processed_dir(dataset)
    num_nodes = infer_num_nodes(proc_dir)

    # Initialise global model
    global_model = SimpleGRU(num_nodes=num_nodes, hidden_size=hidden)
    global_state = global_model.state_dict()

    store_global_model(global_state, dataset, mode, 1)

    # Main FL loop
    for r in range(1, rounds + 1):
        print("\n========== ROUND %d ==========" % r)

        # Select clients
        if mode == "aefl" and r > 1:
            metadata = load_round_metadata(bucket, dataset, mode, r - 1)
            chosen = select_clients_aefl(metadata, ROLES)
        else:
            chosen = select_all_clients(ROLES)

        print("[SERVER] Selected clients:", chosen)

        # Wait for updates
        print("[SERVER] Waiting for updates for round %d..." % r)
        updates = {}
        start_wait = time.time()
        timeout = 300

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    upd = load_client_update(bucket, dataset, mode, r, role)
                    if upd is not None:
                        updates[role] = upd
                        print("[SERVER] Received %s (%d/%d)" %
                              (role, len(updates), len(chosen)))

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout:
                print("[SERVER] WARNING: Timeout waiting for updates.")
                break

            time.sleep(2)

        # Aggregation
        print("[SERVER] Aggregating updates (mode=%s)..." % mode.upper())

        if mode == "aefl":
            global_state = aggregate_aefl(updates)
        elif mode == "fedavg":
            global_state = aggregate_fedavg(updates)
        elif mode == "fedprox":
            global_state = aggregate_fedprox(updates)
        elif mode == "localonly":
            global_state = aggregate_fedavg(updates)
        else:
            global_state = aggregate_fedavg(updates)

        next_round = r + 1
        if next_round <= rounds:
            store_global_model(global_state, dataset, mode, next_round)

    # Final eval
    metrics = evaluate_final_model(
        global_state,
        proc_dir,
        num_nodes,
        hidden,
    )

    for k, v in metrics.items():
        print(" %s = %.6f" % (k, v))

    generate_cloud_summary(metrics, dataset, mode, rounds)

    print("[SERVER] Training finished.")


if __name__ == "__main__":
    main()
