#!/usr/bin/env bash
set -e

# Usage:
#   ./bin/run_all_main_experiments.sh
#
# Runs:
#   (sz|los|pems08) × (localonly, fedavg, fedprox, aefl)

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd "$ROOT_DIR"

DATASETS=("sz" "los" "pems08")

for D in "${DATASETS[@]}"; do
  echo
  echo "##################################################"
  echo "[runner] Running all baselines for dataset=${D}"
  echo "##################################################"
  ./bin/run_baselines_dataset.sh "$D"
done

echo
echo "[runner] All main experiments complete."
echo "[runner] You can now generate plots via: ./bin/generate_all_plots.sh"

