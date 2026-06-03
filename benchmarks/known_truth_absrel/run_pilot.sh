#!/usr/bin/env bash
set -euo pipefail

echo "USER-RUN ONLY: pilot benchmark may take substantial time."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_pilot.yaml"

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/01_simulate_known_truth_dataset.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/02_run_babappa_on_dataset.py" --config "${CONFIG}" --continue-on-failure
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/03_prepare_absrel_inputs.py" --config "${CONFIG}"
