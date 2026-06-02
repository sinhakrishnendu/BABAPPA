#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

OUTROOT="${1:-publication_benchmark/results}"

cd "$(dirname "$0")/../.."

find "$OUTROOT/reference_runs" -mindepth 1 -maxdepth 1 -type d | sort | while read -r family_dir; do
  panel_id="$(basename "$family_dir")"
  echo "Running codeml/HyPhy references for ${panel_id}"
  if command -v codeml >/dev/null 2>&1; then
    (cd "$family_dir/codeml" && bash run_codeml_modelA.sh && bash run_codeml_null.sh)
  else
    echo "codeml unavailable; leaving ${panel_id} codeml pending"
  fi
  if command -v hyphy >/dev/null 2>&1; then
    (cd "$family_dir/hyphy" && bash run_absrel.sh && bash run_meme.sh)
  else
    echo "hyphy unavailable; leaving ${panel_id} HyPhy pending"
  fi
done
