#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_smoke.yaml"

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/01_simulate_known_truth_dataset.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/02_run_babappa_on_dataset.py" --config "${CONFIG}" --continue-on-failure
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/03_prepare_absrel_inputs.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/04_run_absrel.py" --config "${CONFIG}" --parse-only --continue-on-failure
