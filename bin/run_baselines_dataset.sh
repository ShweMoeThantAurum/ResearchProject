#!/usr/bin/env bash
set -e

# Usage:
#   ./bin/run_baselines_dataset.sh <dataset>
#
# Example:
#   ./bin/run_baselines_dataset.sh sz

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

if [ $# -ne 1 ]; then
  echo "Usage: $0 <dataset: sz|los|pems08>"
  exit 1
fi

DATASET_RAW="$1"
DATASET="$(echo "$DATASET_RAW" | tr '[:upper:]' '[:lower:]')"

case "$DATASET" in
  sz|los|pems08) ;;
  *)
    echo "ERROR: dataset must be one of: sz, los, pems08"
    exit 1
    ;;
esac

cd "$ROOT_DIR"

MODES=("localonly" "fedavg" "fedprox" "aefl")

for MODE in "${MODES[@]}"; do
  echo
  echo "=================================================="
  echo "[runner] Running baseline: dataset=${DATASET}, mode=${MODE}"
  echo "=================================================="
  ./bin/run_experiment.sh "$DATASET" "$MODE"
done

echo
echo "[runner] All baselines finished for dataset=${DATASET}"
echo "[runner] Check outputs/summaries/${DATASET}/ and outputs/plots/ after plotting."

