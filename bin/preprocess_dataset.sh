#!/usr/bin/env bash
set -e

# Usage:
#   ./bin/preprocess_dataset.sh <dataset>
#
# Example:
#   ./bin/preprocess_dataset.sh sz
#   ./bin/preprocess_dataset.sh los
#   ./bin/preprocess_dataset.sh pems08

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

echo "==============================================="
echo "[preprocess] Dataset  : $DATASET"
echo "==============================================="

# 1) Preprocess raw → global train/val/test splits
python -m src.fl.data.preprocess "$DATASET"

# 2) Partition global splits → 5 clients
python -m src.fl.data.partition "$DATASET"

echo "[preprocess] Done."
echo "[preprocess] Processed data in: datasets/processed/${DATASET}/"
echo "  - global/train.pt, val.pt, test.pt"
echo "  - roadside.pt, vehicle.pt, sensor.pt, camera.pt, bus.pt"

