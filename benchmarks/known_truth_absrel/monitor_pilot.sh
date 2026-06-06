#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${ROOT_DIR}/benchmark_runs/known_truth_absrel_pilot"

echo "Known-truth aBSREL pilot monitor"
echo "run_dir: ${RUN_DIR}"
echo

if [[ -s "${RUN_DIR}/manifest.tsv" ]]; then
  echo "expected families: $(($(wc -l < "${RUN_DIR}/manifest.tsv") - 1))"
else
  echo "expected families: manifest missing"
fi

if [[ -s "${RUN_DIR}/babappa_results.tsv" ]]; then
  echo "BABAPPA result rows: $(($(wc -l < "${RUN_DIR}/babappa_results.tsv") - 1))"
else
  echo "BABAPPA result rows: 0"
fi

echo "BABAPPA completed family dirs: $(find "${RUN_DIR}/babappa_scores" -name gene_summary.tsv 2>/dev/null | wc -l | tr -d ' ')"
echo "aBSREL JSON files: $(find "${RUN_DIR}/absrel_json" -type f -name '*.json' 2>/dev/null | wc -l | tr -d ' ')"

if [[ -s "${RUN_DIR}/absrel_results.tsv" ]]; then
  echo "aBSREL result rows: $(($(wc -l < "${RUN_DIR}/absrel_results.tsv") - 1))"
else
  echo "aBSREL result rows: 0"
fi

echo "failed BABAPPA families: $(find "${RUN_DIR}/babappa_scores" -name benchmark_family_status.json -exec grep -l '\"status\": \"failed\"' {} + 2>/dev/null | wc -l | tr -d ' ')"
echo "failed aBSREL families: $(find "${RUN_DIR}/absrel_logs" -name absrel_family_status.json -exec grep -l '\"status\": \"failed\"' {} + 2>/dev/null | wc -l | tr -d ' ')"
echo "disk size: $(du -sh "${RUN_DIR}" 2>/dev/null | awk '{print $1}')"
echo
echo "active hyphy processes:"
pgrep -fl hyphy || true
echo
echo "active benchmark python processes:"
pgrep -fl "known_truth_absrel|02_run_babappa|04_run_absrel" || true
echo
echo "macOS vm_stat:"
vm_stat 2>/dev/null | head -n 12 || true
echo
echo "macOS memory_pressure:"
memory_pressure 2>/dev/null | head -n 20 || true
echo
echo "top CPU processes:"
ps -axo pid,pcpu,pmem,comm | sort -k2 -nr | head -n 12 || true
