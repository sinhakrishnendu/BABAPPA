#!/usr/bin/env bash
set -euo pipefail

echo "USER-RUN ONLY: pilot benchmark may take substantial time."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_pilot.yaml"
JOBS="${BABAPPA_BENCH_JOBS:-12}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown)"

echo "jobs: ${JOBS}"
echo "cpu_count: ${CPU_COUNT}"
echo "thread caps: OMP=${OMP_NUM_THREADS:-unset} OPENBLAS=${OPENBLAS_NUM_THREADS:-unset} MKL=${MKL_NUM_THREADS:-unset} VECLIB=${VECLIB_MAXIMUM_THREADS:-unset} NUMEXPR=${NUMEXPR_NUM_THREADS:-unset}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/01_simulate_known_truth_dataset.py" --config "${CONFIG}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/02_run_babappa_on_dataset.py" --config "${CONFIG}" --continue-on-failure --jobs "${JOBS}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/03_prepare_absrel_inputs.py" --config "${CONFIG}" --jobs "${JOBS}"
