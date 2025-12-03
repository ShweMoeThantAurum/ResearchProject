#!/usr/bin/env bash
set -e

# Usage:
#   ./bin/preprocess_all_datasets.sh
#
# Preprocesses and partitions:
#   sz, los, pems08

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd "$ROOT_DIR"

DATASETS=("sz" "los" "pems08")

for D in "${DATASETS[@]}"; do
  echo
  echo "##################################################"
  echo "[preprocess] Running preprocessing for dataset=${D}"
  echo "##################################################"
  ./bin/preprocess_dataset.sh "$D"
done

echo
echo "[preprocess] All datasets preprocessed and partitioned."

