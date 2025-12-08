#!/bin/bash
#
# Collect total energy consumption across all datasets and modes.
#
# For each (dataset, mode) combination, reads:
# outputs/<dataset>/<mode>/energy_summary.json
#
# and writes a consolidated CSV:
# energy_all_datasets.csv
# with columns:
# dataset,mode,variant,total_energy_j
OUT_CSV="energy_all_datasets.csv"
echo "dataset,mode,variant,total_energy_j" > "$OUT_CSV"
for DATASET in sz pems08 los; do
  for MODE in aefl fedavg fedprox; do
    FILE="outputs/${DATASET}/${MODE}/energy_summary.json"
    if [ -f "$FILE" ]; then
      TOTAL=$(jq .total_energy_j "$FILE")
      VARIANT=$(jq -r .variant "$FILE")
      echo "$DATASET,$MODE,$VARIANT,$TOTAL" >> "$OUT_CSV"
    else
      echo "[WARN] Missing energy summary: $FILE"
    fi
  done
done
echo "[OK] Energy CSV generated → $OUT_CSV"
