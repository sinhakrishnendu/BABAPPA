#!/usr/bin/env bash
set -euo pipefail

echo "USER-RUN ONLY: paper benchmark is a long run."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_paper.yaml"

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/01_simulate_known_truth_dataset.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/02_run_babappa_on_dataset.py" --config "${CONFIG}" --continue-on-failure
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/03_prepare_absrel_inputs.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/04_run_absrel.py" --config "${CONFIG}" --continue-on-failure
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/05_compare_against_truth.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/06_make_benchmark_report.py" --config "${CONFIG}"
