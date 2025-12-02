"""
Main orchestration loop for the federated learning server.

Responsibilities:
  - clear S3 prefix for a fresh experiment
  - create and upload the initial global model
  - for each round:
      - choose participating clients
      - wait for their updates
      - aggregate into a new global model
      - upload the next round's global model
  - evaluate the final model
  - save summary artefacts for later analysis
"""

import os
import time

from src.fl.utils.logger import log_event
from src.fl.server.utils_server import (
    get_dataset_name,
    get_proc_dir,
    infer_num_nodes,
    build_initial_model,
)
from src.fl.server.s3_io import (
    get_bucket_name,
    get_round_prefix,
    clear_prefix,
    upload_global_model,
    download_client_update,
    load_round_metadata,
)
from src.fl.server.selection import select_all_clients, select_clients_aefl
from src.fl.server.aggregation import (
    aggregate_fedavg,
    aggregate_fedprox,
    aggregate_aefl,
)
from src.fl.server.evaluate import evaluate_global_model
from src.fl.server.summary import save_summaries


ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def get_fl_mode():
    """
    Read the FL_MODE environment variable and normalise it.

    Supported values:
      - "AEFL"
      - "FedAvg"
      - "FedProx"
      - "LocalOnly"

    Anything else falls back to "AEFL".
    """
    mode = os.environ.get("FL_MODE", "AEFL").strip().lower()
    valid = ["aefl", "fedavg", "fedprox", "localonly"]
    if mode not in valid:
        print("[SERVER] WARNING: invalid FL_MODE '{}', using AEFL".format(mode))
        return "aefl"
    return mode


def get_fl_rounds():
    """
    Read the FL_ROUNDS environment variable and return an integer.

    Defaults to 20 rounds.
    """
    try:
        return int(os.environ.get("FL_ROUNDS", "20"))
    except ValueError:
        return 20


def get_hidden_size():
    """
    Read the HIDDEN_SIZE environment variable for the GRU model.

    Defaults to 64 hidden units.
    """
    try:
        return int(os.environ.get("HIDDEN_SIZE", "64"))
    except ValueError:
        return 64


def run_server():
    """
    Entry point for starting the federated learning server process.

    This function is intended to be run as:
        FL_MODE=AEFL python -m src.fl.server

    or via an explicit import and call in a script.
    """
    mode = get_fl_mode()
    dataset_name = get_dataset_name()
    proc_dir = get_proc_dir(dataset_name)
    rounds = get_fl_rounds()
    hidden_size = get_hidden_size()

    bucket = get_bucket_name()
    prefix = get_round_prefix(dataset_name, mode)

    print(
        "[SERVER] CONFIG dataset={} mode={} rounds={} hidden={}".format(
            dataset_name, mode.upper(), rounds, hidden_size
        )
    )
    print("[SERVER] Using bucket={} prefix={}".format(bucket, prefix))

    # Clean any previous artefacts for this dataset and mode
    clear_prefix(bucket, prefix)

    # Initialise global model
    num_nodes = infer_num_nodes(proc_dir)
    global_state = build_initial_model(num_nodes, hidden_size)

    # Upload global model for round 1
    upload_global_model(bucket, prefix, round_id=1, state_dict=global_state)

    # Federated rounds
    for r in range(1, rounds + 1):
        print("\n========== ROUND {} ==========".format(r))

        if mode == "aefl" and r > 1:
            previous_meta = load_round_metadata(bucket, prefix, r - 1)
            chosen_roles = select_clients_aefl(previous_meta, ROLES)
        else:
            chosen_roles = select_all_clients(ROLES)

        print("[SERVER] Selected clients:", chosen_roles)

        updates = {}
        start_wait = time.time()
        timeout_sec = 300

        print("[SERVER] Waiting for updates for round {}...".format(r))

        while len(updates) < len(chosen_roles):
            for role in chosen_roles:
                if role in updates:
                    continue

                state = download_client_update(bucket, prefix, r, role)
                if state is not None:
                    updates[role] = state
                    print(
                        "[SERVER] Received update from {} ({}/{})".format(
                            role, len(updates), len(chosen_roles)
                        )
                    )

            if len(updates) == len(chosen_roles):
                break

            if time.time() - start_wait > timeout_sec:
                print("[SERVER] WARNING: timeout waiting for client updates.")
                break

            time.sleep(2.0)

        # Aggregation
        start_aggr = time.time()

        if mode == "aefl":
            global_state = aggregate_aefl(updates)
            aggr_label = "AEFL"
        elif mode == "fedavg":
            global_state = aggregate_fedavg(updates)
            aggr_label = "FedAvg"
        elif mode == "fedprox":
            global_state = aggregate_fedprox(updates)
            aggr_label = "FedProx"
        elif mode == "localonly":
            # LocalOnly still aggregates the uploaded models so that
            # evaluation can be carried out on a single global model.
            global_state = aggregate_fedavg(updates)
            aggr_label = "LocalOnly"
        else:
            global_state = aggregate_fedavg(updates)
            aggr_label = "FedAvgCompat"

        aggr_time = time.time() - start_aggr
        print(
            "[SERVER] Aggregation complete mode={} time={:.3f}s".format(
                aggr_label, aggr_time
            )
        )

        log_event(
            "server_aggregation.log",
            {
                "round": r,
                "mode": aggr_label,
                "aggregation_time_sec": aggr_time,
                "num_updates": len(updates),
            },
        )

        # Upload global model for the next round
        next_round = r + 1
        if next_round <= rounds:
            upload_global_model(bucket, prefix, next_round, global_state)

    # Final evaluation
    metrics = evaluate_global_model(
        global_state,
        proc_dir=proc_dir,
        num_nodes=num_nodes,
        hidden_size=hidden_size,
    )

    print("\n[SERVER] Final Evaluation Metrics:")
    for k, v in metrics.items():
        print(" {} = {:.6f}".format(k, v))

    log_event(
        "server_final_metrics.log",
        {
            "dataset": dataset_name,
            "mode": mode.upper(),
            "rounds": rounds,
            "metrics": metrics,
        },
    )

    save_summaries(
        final_metrics=metrics,
        num_rounds=rounds,
        dataset_name=dataset_name,
        mode_name=mode,
    )

    print("[SERVER] Training finished after {} rounds.".format(rounds))


if __name__ == "__main__":
    run_server()
