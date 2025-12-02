"""
Main server orchestration loop for federated learning.

Flow:
    - Load config and dataset info
    - Initialise global model
    - For each round:
        * select clients
        * wait for updates
        * aggregate
        * store model
    - Perform final evaluation
    - Write summary to outputs + S3
"""

import os
import time
import torch

from src.fl.server.selection import select_all_clients, select_clients_aefl
from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)
from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import generate_cloud_summary
from src.fl.server.s3_io import upload_json, download_json, upload_bytes, download_bytes
from src.fl.server.utils_server import (
    get_proc_dir,
    infer_num_nodes,
    get_hidden_size,
    get_fl_rounds,
    init_global_model,
    store_global_model,
)


ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def main():
    dataset = os.environ.get("DATASET")
    mode = os.environ.get("FL_MODE", "AEFL")
    bucket = os.environ.get("S3_BUCKET", "aefl")

    proc_dir = get_proc_dir(dataset)
    hidden = get_hidden_size()
    rounds = get_fl_rounds()

    print("[SERVER] CONFIG: dataset=%s mode=%s rounds=%d hidden=%d" %
          (dataset, mode, rounds, hidden))

    num_nodes = infer_num_nodes(proc_dir)
    global_state = init_global_model(num_nodes, hidden)

    # Store initial model
    store_global_model(global_state, 1, dataset, mode)

    # FL rounds
    for r in range(1, rounds + 1):
        print("\n========== ROUND %d ==========" % r)

        if mode.lower() == "aefl" and r > 1:
            prev_meta = download_json(bucket, dataset, mode, r - 1, "metadata.json")
            chosen = select_clients_aefl(prev_meta, ROLES)
        else:
            chosen = select_all_clients(ROLES)

        print("[SERVER] Selected clients:", chosen)

        updates = {}
        start_wait = time.time()
        timeout = 300

        print("[SERVER] Waiting for updates for round %d..." % r)

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    blob = download_bytes(bucket, dataset, mode, r, "%s_update" % role)
                    if blob is not None:
                        updates[role] = torch.load(
                            os.path.join("/tmp", "%s_r%d.pth" % (role, r)),
                            map_location="cpu"
                        )

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout:
                print("[SERVER] WARNING: Timeout waiting for updates.")
                break

            time.sleep(2)

        # Aggregation
        if mode.lower() == "aefl":
            global_state = aggregate_aefl(updates)
        elif mode.lower() == "fedavg":
            global_state = aggregate_fedavg(updates)
        elif mode.lower() == "fedprox":
            global_state = aggregate_fedprox(updates)
        elif mode.lower() == "localonly":
            global_state = aggregate_fedavg(updates)
        else:
            global_state = aggregate_fedavg(updates)

        next_round = r + 1
        if next_round <= rounds:
            store_global_model(global_state, next_round, dataset, mode)

    # Final evaluation
    metrics = evaluate_final_model(global_state, proc_dir, num_nodes, hidden)

    print("\n[SERVER] Final Evaluation:")
    for k, v in metrics.items():
        print(" %s = %.6f" % (k, v))

    generate_cloud_summary(metrics, rounds, mode, dataset)

    print("[SERVER] Training finished after %d rounds." % rounds)


if __name__ == "__main__":
    main()
