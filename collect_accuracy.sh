#!/bin/bash
#
# Collect final accuracy metrics across all datasets and modes.
#
# For each (dataset, mode) combination, reads:
# outputs/<dataset>/<mode>/final_metrics_<mode>.json
#
# and writes a consolidated CSV:
# accuracy_all_datasets.csv
# with columns:
# dataset,mode,variant,MAE,RMSE,MAPE
OUT_CSV="accuracy_all_datasets.csv"
echo "dataset,mode,variant,MAE,RMSE,MAPE" > "$OUT_CSV"
for DATASET in sz pems08 los; do
  for MODE in fedavg fedprox aefl; do
    METRICS_FILE="outputs/${DATASET}/${MODE}/final_metrics_${MODE}.json"
    if [ -f "$METRICS_FILE" ]; then
      MAE=$(jq .MAE "$METRICS_FILE")
      RMSE=$(jq .RMSE "$METRICS_FILE")
      MAPE=$(jq .MAPE "$METRICS_FILE")
      VARIANT=$(jq -r .variant "$METRICS_FILE")
      echo "${DATASET},${MODE},${VARIANT},${MAE},${RMSE},${MAPE}" >> "$OUT_CSV"
    else
      echo "[WARN] Missing metrics file: $METRICS_FILE"
    fi
  done
done
echo "[OK] Accuracy CSV generated → $OUT_CSV"
