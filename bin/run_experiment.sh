#!/usr/bin/env bash
set -e

# Usage:
#   ./bin/run_experiment.sh <dataset> <mode>
#
# Examples:
#   ./bin/run_experiment.sh sz aefl
#   ./bin/run_experiment.sh los fedavg
#   ./bin/run_experiment.sh pems08 fedprox
#
# Modes (case-insensitive): aefl, fedavg, fedprox, localonly
# The corresponding YAML config is chosen automatically based on mode.

# Resolve repo root (works even if called from another directory)
SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

if [ $# -ne 2 ]; then
  echo "Usage: $0 <dataset: sz|los|pems08> <mode: aefl|fedavg|fedprox|localonly>"
  exit 1
fi

DATASET_RAW="$1"
MODE_RAW="$2"

DATASET="$(echo "$DATASET_RAW" | tr '[:upper:]' '[:lower:]')"
MODE_LOWER="$(echo "$MODE_RAW" | tr '[:upper:]' '[:lower:]')"
MODE_UPPER="$(echo "$MODE_RAW" | tr '[:lower:]' '[:upper:]')"

case "$DATASET" in
  sz|los|pems08) ;;
  *)
    echo "ERROR: dataset must be one of: sz, los, pems08"
    exit 1
    ;;
esac

case "$MODE_LOWER" in
  aefl|fedavg|fedprox|localonly) ;;
  *)
    echo "ERROR: mode must be one of: aefl, fedavg, fedprox, localonly"
    exit 1
    ;;
esac

cd "$ROOT_DIR"

# Core env for this run (YAML overrides hyperparams internally)
export DATASET="$DATASET"
export FL_MODE="$MODE_UPPER"

# Optional: default S3 + rounds if not set from outside
: "${AWS_REGION:=us-east-1}"
: "${S3_BUCKET:=aefl}"
: "${FL_ROUNDS:=20}"

export AWS_REGION
export S3_BUCKET
export FL_ROUNDS

echo "[runner] Root dir : $ROOT_DIR"
echo "[runner] Dataset  : $DATASET"
echo "[runner] Mode     : $MODE_UPPER"
echo "[runner] S3 bucket: $S3_BUCKET"
echo "[runner] FL rounds: $FL_ROUNDS"
echo "[runner] Using mode-default YAML config (aefl_default / fedavg / fedprox / localonly)"

# 1) Start clients with docker-compose (detached)
echo "[runner] Starting FL clients via docker compose..."
(
  cd "$ROOT_DIR/docker"
  docker compose up --build -d
)

# Small delay to let clients boot and wait for round 1 global model
sleep 5

# 2) Run server (blocking)
echo "[runner] Starting FL server..."
python -m src.fl.server

# 3) After server finishes, gather logs and stop clients
echo "[runner] Server finished. Stopping FL clients..."
(
  cd "$ROOT_DIR/docker"
  # Optional: capture client logs for this run
  mkdir -p "$ROOT_DIR/outputs/logs"
  docker compose logs --no-color > "$ROOT_DIR/outputs/logs/docker_${DATASET}_${MODE_LOWER}.log" 2>&1 || true

  docker compose down
)

echo "[runner] Done."
echo "[runner] Metrics: outputs/summaries/${DATASET}/${MODE_LOWER}/"
echo "[runner] Logs   : outputs/logs/"

