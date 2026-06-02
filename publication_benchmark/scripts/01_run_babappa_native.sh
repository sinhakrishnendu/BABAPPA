#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

MANIFEST="${1:-publication_benchmark/panel_template.tsv}"
OUTROOT="${2:-publication_benchmark/results}"
CONFIG="${CONFIG:-publication_benchmark/benchmark_config.env}"

cd "$(dirname "$0")/../.."
source "$CONFIG"
mkdir -p "$OUTROOT/babappa" "$OUTROOT/logs"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r panel_id gene_family cds_msa tree_file foreground expected_category notes; do
  if [[ -z "${panel_id}" || "${panel_id}" == \#* ]]; then
    continue
  fi
  echo "Running BABAPPA-native benchmark for ${panel_id}"
  babappa predict-branch-sites \
    --msa "$cds_msa" \
    --tree "$tree_file" \
    --foreground "$foreground" \
    --model-package "$BABAPPA_MODEL_PACKAGE" \
    --outdir "$OUTROOT/babappa/$panel_id" \
    --device "$BABAPPA_DEVICE" \
    --null-replicates "$BABAPPA_NULL_REPLICATES" \
    --null-seed "$BABAPPA_NULL_SEED" \
    2>&1 | tee "$OUTROOT/logs/${panel_id}.babappa.log"
done
