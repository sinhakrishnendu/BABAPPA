#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

ORTHOFINDER_ROOT="${1:-publication_benchmark/drosophila_orthofinder}"
BENCHROOT="${2:-publication_benchmark/drosophila_absrel_benchmark}"
MAX_FAMILIES="${MAX_FAMILIES:-50}"
MIN_TAXA="${MIN_TAXA:-6}"
MIN_CODONS="${MIN_CODONS:-100}"
MAX_CODONS="${MAX_CODONS:-1500}"
MAX_GAP_FRACTION="${MAX_GAP_FRACTION:-0.25}"
MAX_MEAN_PDISTANCE="${MAX_MEAN_PDISTANCE:-0.25}"

cd "$(dirname "$0")/../.."

python publication_benchmark/scripts/drosophila_02_extract_single_copy_orthologs.py \
  --orthofinder-root "$ORTHOFINDER_ROOT" \
  --prepared-dir publication_benchmark/drosophila_orthofinder \
  --outdir "$BENCHROOT/single_copy_orthologs" \
  --max-orthogroups "$MAX_FAMILIES"

python publication_benchmark/scripts/drosophila_03_build_babappa_absrel_inputs.py \
  --orthofinder-root "$ORTHOFINDER_ROOT" \
  --prepared-dir publication_benchmark/drosophila_orthofinder \
  --outdir "$BENCHROOT" \
  --max-families "$MAX_FAMILIES" \
  --min-taxa "$MIN_TAXA" \
  --min-codons "$MIN_CODONS" \
  --max-codons "$MAX_CODONS" \
  --max-gap-fraction "$MAX_GAP_FRACTION" \
  --max-mean-pdistance "$MAX_MEAN_PDISTANCE" \
  --foreground leaves

bash publication_benchmark/scripts/drosophila_04_run_babappa_absrel_user.sh \
  "$BENCHROOT/drosophila_babappa_absrel_panel.tsv" \
  "$BENCHROOT/results"

python publication_benchmark/scripts/drosophila_05_summarize_babappa_absrel.py \
  --panel "$BENCHROOT/drosophila_babappa_absrel_panel.tsv" \
  --results-root "$BENCHROOT/results" \
  --outdir "$BENCHROOT/results/summary"

echo "Done. Summary:"
echo "$BENCHROOT/results/summary/babappa_vs_hyphy_absrel_summary.md"
