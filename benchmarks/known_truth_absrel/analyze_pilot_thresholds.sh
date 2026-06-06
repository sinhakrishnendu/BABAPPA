#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_pilot.yaml"
RUN_DIR="${ROOT_DIR}/benchmark_runs/known_truth_absrel_pilot"

if [[ ! -s "${RUN_DIR}/method_comparison.tsv" ]]; then
  echo "Missing pilot method comparison: ${RUN_DIR}/method_comparison.tsv"
  echo "Run the pilot BABAPPA/aBSREL benchmark first."
  exit 1
fi

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/08_threshold_sweep.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/09_compare_operating_points.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/06_make_benchmark_report.py" --config "${CONFIG}"
