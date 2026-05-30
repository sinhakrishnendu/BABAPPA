#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'

# OOD-aware family build for WRKY_candidate_02_close.
# Steps: download -> BLASTP best hit -> recover CDS -> sanitize -> align -> tree -> prefilter -> gated import.
MAX_MEAN_PDISTANCE=0.35
MIN_TAXA=6
MIN_CODONS=100
FAMILY_ID=WRKY_candidate_02_close
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
cd "${SCRIPT_DIR}"
PREFILTER_DIR="${WORKSPACE_DIR}/prefilter/${FAMILY_ID}"

echo "Fill the acquisition scripts with exact data sources before running this workflow."
bash download_ensembl_proteome_cds.sh
bash run_blastp_best_hit.sh
bash recover_cds_from_best_hits.sh
bash build_tree.sh
babappa prefilter-empirical-family \
  --cds-fasta candidate.cds.fasta \
  --tree-file tree.treefile \
  --foreground Arabidopsis_thaliana \
  --outdir "${PREFILTER_DIR}" \
  --max-mean-pdistance "${MAX_MEAN_PDISTANCE}" \
  --min-taxa "${MIN_TAXA}" \
  --min-codons "${MIN_CODONS}"
decision=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["decision"])' "${PREFILTER_DIR}/empirical_family_prefilter.json")
case "${decision}" in
  accept|accept_with_caution)
    babappa add-prefiltered-family-to-pilot \
      --workspace "${WORKSPACE_DIR}" \
      --prefilter-dir "${PREFILTER_DIR}" \
      --panel-id "${FAMILY_ID}" \
      --expected-category likely_positive \
      --reference-status planned
    ;;
  *)
    echo "Not importing ${FAMILY_ID}; prefilter decision was ${decision}."
    exit 2
    ;;
esac
