#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

ROOT="${1:-publication_benchmark/drosophila_orthofinder}"
INPUT_DIR="${ROOT}/orthofinder_input"
OUT_PARENT="${ORTHOFINDER_OUT:-${ROOT}/orthofinder_run_$(date +%Y%m%d_%H%M%S)}"
THREADS="${ORTHOFINDER_THREADS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 8)}"

if ! command -v orthofinder >/dev/null 2>&1; then
  echo "ERROR: orthofinder was not found on PATH." >&2
  echo "Install in molevo with:" >&2
  echo "  conda install -y -c bioconda -c conda-forge orthofinder diamond mafft fasttree" >&2
  exit 127
fi

if [ ! -d "${INPUT_DIR}" ]; then
  echo "ERROR: missing OrthoFinder input directory: ${INPUT_DIR}" >&2
  echo "Run drosophila_00_prepare_orthofinder_inputs.py first." >&2
  exit 2
fi

echo "Running OrthoFinder"
echo "  input: ${INPUT_DIR}"
echo "  output directory: ${OUT_PARENT}"
echo "  threads: ${THREADS}"

if [ -e "${OUT_PARENT}" ]; then
  echo "ERROR: OrthoFinder output path already exists: ${OUT_PARENT}" >&2
  echo "Set ORTHOFINDER_OUT to a fresh path or remove the existing empty directory after review." >&2
  exit 2
fi

orthofinder \
  -f "${INPUT_DIR}" \
  -S diamond \
  -t "${THREADS}" \
  -a "${THREADS}" \
  -o "${OUT_PARENT}"

echo "OrthoFinder completed. Results should be under:"
RESULTS="$(find "${OUT_PARENT}" -maxdepth 2 -type d -name 'Results_*' -print | sort | tail -n 1 || true)"
if [ -z "${RESULTS}" ]; then
  echo "ERROR: OrthoFinder returned but no Results_* directory was found under ${OUT_PARENT}" >&2
  exit 3
fi
echo "${RESULTS}"
