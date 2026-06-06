#!/usr/bin/env bash
set -euo pipefail

echo "USER-RUN ONLY: paper comparison uses frozen pilot-selected threshold policy."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_paper.yaml"
POLICY="${ROOT_DIR}/benchmarks/known_truth_absrel/threshold_policy.yaml"

if [[ ! -s "${POLICY}" ]]; then
  echo "Freeze threshold policy from pilot before paper benchmark."
  exit 1
fi

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/05_compare_against_truth.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/validate_result_tables.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/10_apply_frozen_threshold_policy.py" --config "${CONFIG}" --threshold-policy "${POLICY}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/06_make_benchmark_report.py" --config "${CONFIG}"
