#!/bin/bash
#
# Run baseline FL experiments (no DP, no compression) for:
#   Datasets: SZ, PeMS08, LOS
#   Modes   : FedAvg, FedProx, AEFL
#
# For each (dataset, mode) pair this script:
#   - sets environment variables
#   - starts client containers via docker compose
#   - runs the Python server orchestrator
#   - shuts down containers

DATASETS=("sz" "pems08" "los")
MODES=("FedAvg" "FedProx" "AEFL")

for DATASET in "${DATASETS[@]}"; do
  for MODE in "${MODES[@]}"; do

    echo "==============================="
    echo " Running BASELINE: DATASET=$DATASET MODE=$MODE"
    echo "==============================="

    export DATASET="$DATASET"
    export FL_MODE="$MODE"

    # Baseline → no variants, no DP, no compression
    export VARIANT_ID=""
    export DP_ENABLED=false
    export COMPRESSION_ENABLED=false

    # Optional: clean logs before each run
    rm -f run_logs/*.log

    # Start clients in the background
    docker compose up -d

    # Run server orchestration (blocks until training completes)
    python -m src.fl.server.main

    # Stop clients
    docker compose down

    echo "Finished run: ${DATASET}-${MODE}"
    echo ""
  done
done
