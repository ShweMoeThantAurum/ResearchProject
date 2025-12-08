#!/bin/bash
#
# Aggregate accuracy and energy results for DP ablation runs.
#
# Reads:
# outputs/<dataset>/aefl/final_metrics_aefl_dp_sigma_<sigma>.json
# outputs/<dataset>/aefl/energy_summary_dp_sigma_<sigma>.json
#
# Writes a CSV:
# privacy_dp_results.csv
# with columns:
# dataset,sigma,MAE,RMSE,MAPE,total_energy_j
OUT="privacy_dp_results.csv"
# Header row
echo "dataset,sigma,MAE,RMSE,MAPE,total_energy_j" > "$OUT"
DATASETS=("sz" "pems08" "los")
SIGMAS=("0.0" "0.01" "0.05" "0.10" "0.20")
for DATASET in "${DATASETS[@]}"; do
  for SIGMA in "${SIGMAS[@]}"; do
    MODE_DIR="outputs/${DATASET}/aefl"
    METRICS="${MODE_DIR}/final_metrics_aefl_dp_sigma_${SIGMA}.json"
    ENERGY="${MODE_DIR}/energy_summary_dp_sigma_${SIGMA}.json"
    if [ ! -f "$METRICS" ]; then
      echo "[WARN] Missing metrics file: $METRICS"
      continue
    fi
    # Accuracy metrics from final_metrics JSON
    MAE=$(jq .MAE "$METRICS")
    RMSE=$(jq .RMSE "$METRICS")
    MAPE=$(jq .MAPE "$METRICS")
    # Energy from variant-specific energy summary (if available)
    if [ -f "$ENERGY" ]; then
      TOTAL=$(jq .total_energy_j "$ENERGY")
    else
      echo "[WARN] Missing energy file: $ENERGY (using 0)"
      TOTAL="0"
    fi
    echo "$DATASET,$SIGMA,$MAE,$RMSE,$MAPE,$TOTAL" >> "$OUT"
  done
done
echo "[OK] DP CSV saved → $OUT"
