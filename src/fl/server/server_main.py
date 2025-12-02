"""
Main server loop for federated learning.
Handles:
- loading global state
- selecting clients (FedAvg, FedProx, AEFL)
- collecting updates
- aggregation
- evaluation
- saving summaries
"""

import os
import json
import time

import torch

from src.fl.config import settings
from src.fl.models import SimpleGRU
from src.fl.server.selection import select_all_clients, select_clients_aefl
from src.fl.server.aggregation import aggregate_fedavg, aggregate_fedprox, aggregate_aefl
from src.fl.server.evaluate import evaluate_final_model
from src.fl.server.summary import write_summary
from src.fl.server.s3_io import load_update, load_round_metadata, save_global_state
from src.fl.server.utils_server import init_global_model, ensure_dirs


ROLES = ["roadside", "vehicle", "sensor", "camera", "bus"]


def main():
    mode = settings.get_fl_mode()
    rounds = settings.get_fl_rounds()
    hidden = settings.get_hidden_size()

    print("[SERVER] Starting | mode={} rounds={} hidden={}".format(mode, rounds, hidden))

    ensure_dirs()

    # Initialise global model
    global_state = init_global_model(hidden)

    # Save initial global model (round 1)
    save_global_state(global_state, 1)

    # FL rounds
    for r in range(1, rounds + 1):
        print("\n========== ROUND {} ==========".format(r))

        # -----------------------
        # Client selection
        # -----------------------
        if mode == "AEFL" and r > 1:
            metadata = load_round_metadata(r - 1)
            chosen = select_clients_aefl(metadata, ROLES)
        else:
            chosen = select_all_clients(ROLES)

        print("[SERVER] Selected clients:", chosen)

        # -----------------------
        # Collect updates
        # -----------------------
        updates = {}
        start_wait = time.time()
        timeout = 300

        print("[SERVER] Waiting for updates...")

        while len(updates) < len(chosen):
            for role in chosen:
                if role not in updates:
                    upd = load_update(role, r)
                    if upd is not None:
                        updates[role] = upd
                        print("[SERVER] Received {} ({}/{})".format(
                            role, len(updates), len(chosen)
                        ))

            if len(updates) == len(chosen):
                break

            if time.time() - start_wait > timeout:
                print("[SERVER] WARNING: Timeout waiting for updates")
                break

            time.sleep(1)

        # -----------------------
        # Aggregation
        # -----------------------
        if mode == "AEFL":
            global_state = aggregate_aefl(updates)
        elif mode == "FedAvg":
            global_state = aggregate_fedavg(updates)
        elif mode == "FedProx":
            global_state = aggregate_fedprox(updates)
        elif mode == "LocalOnly":
            global_state = aggregate_fedavg(updates)
        else:
            global_state = aggregate_fedavg(updates)

        # Save global model for next round
        if r < rounds:
            save_global_state(global_state, r + 1)

    # -----------------------
    # Final evaluation
    # -----------------------
    metrics = evaluate_final_model(global_state)

    print("\n[SERVER] Final Evaluation:")
    for k, v in metrics.items():
        print(" {} = {:.6f}".format(k, v))

    write_summary(metrics, mode)

    print("[SERVER] Training complete.")
