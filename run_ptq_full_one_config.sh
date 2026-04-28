#!/usr/bin/env bash
set -euo pipefail

# Runs static INT8 PTQ on the real Places365 validation set (no --smoke),
# but only for a single config to keep runtime reasonable.
#
# Outputs:
#   results/ptq_full_one/ptq_grid.json
#   results/ptq_full_one/report.md
#
# Dataset discovery:
# - Uses $PLACES365_ROOT if set.
# - Otherwise falls back to the historical local path used in this workspace.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

python_bin="$repo_root/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo "[error] Missing venv python at: $python_bin"
  echo "Create it with: python3.11 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt"
  exit 1
fi

default_data_root="/Users/queca/Library/Mobile Documents/com~apple~CloudDocs/cv project/scene-classification-ptq_old/places365_data"
data_root="${PLACES365_ROOT:-$default_data_root}"

if [[ ! -d "$data_root" ]]; then
  echo "[error] Places365 data root not found: $data_root"
  echo "Set PLACES365_ROOT to your dataset path, e.g.:"
  echo "  PLACES365_ROOT=\"/path/to/places365_data\" ./run_ptq_full_one_config.sh"
  exit 1
fi

results_dir="results/ptq_full_one"

exec "$python_bin" -m src.experiments.run_ptq_grid \
  --data-root "$data_root" \
  --weight-modes per_channel \
  --calibration-batches 100 \
  --results-dir "$results_dir"
