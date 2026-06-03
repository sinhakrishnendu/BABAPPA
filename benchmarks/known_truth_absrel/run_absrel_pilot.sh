#!/usr/bin/env bash
set -euo pipefail

echo "USER-RUN ONLY: aBSREL pilot comparator may take substantial time."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_pilot.yaml"

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/04_run_absrel.py" --config "${CONFIG}" --continue-on-failure
