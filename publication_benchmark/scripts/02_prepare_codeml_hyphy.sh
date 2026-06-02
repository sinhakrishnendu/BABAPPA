#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - REVIEW BEFORE RUNNING"

MANIFEST="${1:-publication_benchmark/panel_template.tsv}"
OUTROOT="${2:-publication_benchmark/results}"

cd "$(dirname "$0")/../.."
mkdir -p "$OUTROOT/reference_runs"

tail -n +2 "$MANIFEST" | while IFS=$'\t' read -r panel_id gene_family cds_msa tree_file foreground expected_category notes; do
  if [[ -z "${panel_id}" || "${panel_id}" == \#* ]]; then
    continue
  fi
  echo "Preparing codeml/HyPhy reference templates for ${panel_id}"
  babappa prepare-codeml-reference \
    --cds-fasta "$cds_msa" \
    --tree "$tree_file" \
    --foreground "$foreground" \
    --outdir "$OUTROOT/reference_runs/$panel_id/codeml"
  babappa prepare-hyphy-reference \
    --cds-fasta "$cds_msa" \
    --tree "$tree_file" \
    --foreground "$foreground" \
    --outdir "$OUTROOT/reference_runs/$panel_id/hyphy"
done
