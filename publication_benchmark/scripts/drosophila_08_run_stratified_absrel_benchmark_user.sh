#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

ORTHOFINDER_ROOT="${1:-publication_benchmark/drosophila_orthofinder}"
BENCHROOT="${2:-publication_benchmark/drosophila_absrel_benchmark_stratified}"
FAMILIES_PER_STRATUM="${FAMILIES_PER_STRATUM:-20}"
MIN_TAXA="${MIN_TAXA:-6}"
MIN_CODONS="${MIN_CODONS:-100}"
MAX_CODONS="${MAX_CODONS:-1500}"

cd "$(dirname "$0")/../.."

python publication_benchmark/scripts/drosophila_07_build_stratified_absrel_panels.py \
  --orthofinder-root "$ORTHOFINDER_ROOT" \
  --prepared-dir publication_benchmark/drosophila_orthofinder \
  --outdir "$BENCHROOT" \
  --families-per-stratum "$FAMILIES_PER_STRATUM" \
  --min-taxa "$MIN_TAXA" \
  --min-codons "$MIN_CODONS" \
  --max-codons "$MAX_CODONS" \
  --foreground leaves

bash publication_benchmark/scripts/drosophila_04_run_babappa_absrel_user.sh \
  "$BENCHROOT/stratified_drosophila_babappa_absrel_panel.tsv" \
  "$BENCHROOT/results"

python publication_benchmark/scripts/drosophila_05_summarize_babappa_absrel.py \
  --panel "$BENCHROOT/stratified_drosophila_babappa_absrel_panel.tsv" \
  --results-root "$BENCHROOT/results" \
  --outdir "$BENCHROOT/results/summary"

echo "Done. Summary:"
echo "$BENCHROOT/results/summary/babappa_vs_hyphy_absrel_summary.md"
