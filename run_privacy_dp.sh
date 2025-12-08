#!/bin/bash
#
# Run AEFL + Differential Privacy experiments for multiple datasets
# and noise levels (σ). Each configuration sets:
#   FL_MODE=aefl
#   DP_ENABLED=true
#   VARIANT_ID=dp_sigma_<sigma>
#
# Results are written under:
#   outputs/<dataset>/aefl/
# and later aggregated by collect_privacy_dp.sh.

DATASETS=("sz" "pems08" "los")
SIGMAS=("0.0" "0.01" "0.05" "0.10" "0.20")

export FL_MODE="aefl"
export COMPRESSION_ENABLED=false
export DP_ENABLED=true

for DATASET in "${DATASETS[@]}"; do
  for SIGMA in "${SIGMAS[@]}"; do

    echo "============================================"
    echo " RUNNING DP EXPERIMENT | DATASET=${DATASET} | SIGMA=${SIGMA}"
    echo "============================================"

    export DATASET="$DATASET"
    export DP_SIGMA="$SIGMA"
    export VARIANT_ID="dp_sigma_${SIGMA}"

    # Clean logs for a fresh run of this (dataset, sigma)
    rm -f run_logs/*.log

    # Restart client containers
    docker compose down -v
    docker compose up -d

    # Run server orchestration (this will also generate cloud summaries
    # including variant-specific final_metrics and energy summaries)
    python -m src.fl.server.main

    # Shutdown clients cleanly
    docker compose down -v

  done
done
