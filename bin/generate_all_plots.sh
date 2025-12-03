#!/usr/bin/env bash
set -e

# Usage:
#   ./bin/generate_all_plots.sh
#
# Generates accuracy, energy, and privacy plots for all datasets/modes
# using the summaries in outputs/summaries/.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
ROOT_DIR="$( cd "${SCRIPT_DIR}/.." && pwd )"

cd "$ROOT_DIR"

echo "[runner] Generating all plots from outputs/summaries/..."
python -m src.fl.analysis.generate_all_plots

echo "[runner] Plots written under outputs/plots/"

