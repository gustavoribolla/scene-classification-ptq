#!/usr/bin/env bash
set -euo pipefail

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

exec "$python_bin" -m src.experiments.run_quantized_demo \
  --data-root "$data_root" \
  --results-dir results/quantized_demo \
  --weight-mode per_channel \
  --calibration-batches 100 \
  --batch-size 32 \
  --sample-count 8 \
  "$@"
