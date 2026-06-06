#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_paper.yaml"
POLICY="${ROOT_DIR}/benchmarks/known_truth_absrel/threshold_policy.yaml"

if [[ ! -s "${POLICY}" ]]; then
  echo "Freeze threshold policy from pilot before paper benchmark."
  exit 1
fi

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/10_apply_frozen_threshold_policy.py" --config "${CONFIG}" --threshold-policy "${POLICY}"
