#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_pilot.yaml"

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/05_compare_against_truth.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/06_make_benchmark_report.py" --config "${CONFIG}"
