#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_smoke.yaml"

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/07_diagnose_run.py" --config "${CONFIG}"
