#!/usr/bin/env bash
set -euo pipefail

echo "USER-RUN ONLY — DO NOT EXECUTE IN CODEX"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG="${ROOT_DIR}/benchmarks/known_truth_absrel/config_smoke.yaml"
RUN_DIR="${ROOT_DIR}/benchmark_runs/known_truth_absrel_smoke"
JOBS="${BABAPPA_BENCH_JOBS:-4}"
CPU_COUNT="$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo unknown)"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "jobs: ${JOBS}"
echo "cpu_count: ${CPU_COUNT}"
echo "thread caps: OMP=${OMP_NUM_THREADS} OPENBLAS=${OPENBLAS_NUM_THREADS} MKL=${MKL_NUM_THREADS} VECLIB=${VECLIB_MAXIMUM_THREADS} NUMEXPR=${NUMEXPR_NUM_THREADS}"

if [[ ! -d "${RUN_DIR}" ]]; then
  echo "Missing smoke run directory: ${RUN_DIR}"
  echo "Run first: bash benchmarks/known_truth_absrel/run_smoke.sh"
  exit 1
fi

if [[ ! -s "${RUN_DIR}/truth/family_truth.tsv" || ! -s "${RUN_DIR}/manifest.tsv" ]]; then
  echo "Smoke truth/manifest files are missing."
  echo "Run first: bash benchmarks/known_truth_absrel/run_smoke.sh"
  exit 1
fi

if ! command -v hyphy >/dev/null 2>&1; then
  echo "HyPhy is not available on PATH. Activate the environment that provides hyphy, then rerun this script."
  exit 1
fi

if [[ ! -s "${RUN_DIR}/absrel_input_manifest.tsv" ]]; then
  python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/03_prepare_absrel_inputs.py" --config "${CONFIG}" --jobs "${JOBS}"
fi

python "${ROOT_DIR}/scripts/benchmarks/known_truth_absrel/04_run_absrel.py" --config "${CONFIG}" --continue-on-failure --jobs "${JOBS}"
