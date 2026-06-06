#!/usr/bin/env bash
set -euo pipefail

echo "Independent aBSREL validation comparator is a long run."
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_validation.yaml"
JOBS="${BABAPPA_BENCH_JOBS:-14}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "config: ${CONFIG}"
echo "jobs: ${JOBS}"
echo "cpu_count: ${CPU_COUNT}"
echo "thread caps: OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS} VECLIB=${VECLIB_MAXIMUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"
python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/04_run_absrel.py" --config "${CONFIG}" --continue-on-failure --jobs "${JOBS}"
