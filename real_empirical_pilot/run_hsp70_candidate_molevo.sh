#!/usr/bin/env bash
set -euo pipefail

echo "MANUAL EXECUTION SCRIPT - HSP70_candidate_01 paralogy/alignment-sensitive diagnostic run"
cd /Users/krishnendu/Documents/GitHub/BABAPPA

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniforge3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/miniforge3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh. Please activate molevo manually and rerun this script." >&2
  exit 2
fi

set +u
conda activate molevo
set -u

export PYTORCH_ENABLE_MPS_FALLBACK="${PYTORCH_ENABLE_MPS_FALLBACK:-1}"
export BABAPPA_BABAPPALIGN_DEVICE="${BABAPPA_BABAPPALIGN_DEVICE:-mps}"
export BABAPPA_BABAPPALIGN_BACKEND="${BABAPPA_BABAPPALIGN_BACKEND:-embedded}"
export BABAPPA_BABAPPALIGN_WORKERS="${BABAPPA_BABAPPALIGN_WORKERS:-2}"
export BABAPPA_BABAPPALIGN_MAX_WORKERS="${BABAPPA_BABAPPALIGN_MAX_WORKERS:-2}"
export BABAPPA_HSP70_DEVICE="${BABAPPA_HSP70_DEVICE:-auto}"

echo "Checking required commands..."
command -v babappa
command -v mafft
command -v babappalign
command -v muscle

babappa validate-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/hsp70_candidate_panel.tsv \
  --outdir real_empirical_pilot/panel_validation_hsp70_candidate_molevo

babappa prefilter-empirical-family \
  --cds-fasta real_empirical_pilot/input/cds/HSP70_candidate_01.cds.fasta \
  --tree-file real_empirical_pilot/input/trees/HSP70_candidate_01.treefile \
  --foreground 'Cucsa.010680|Cucsa.010680.1' \
  --outdir real_empirical_pilot/prefilter/HSP70_candidate_01_molevo

echo "NOTE: The prefilter currently rejects this family as possible paralogy/high divergence."
echo "Continuing only as a diagnostic software run, not as empirical evidence."

babappa run-empirical-pilot-panel \
  --panel-manifest real_empirical_pilot/manifest/hsp70_candidate_panel.tsv \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir real_empirical_pilot/babappa_run_hsp70_candidate \
  --methods identity,mafft,babappalign,muscle \
  --device "$BABAPPA_HSP70_DEVICE" \
  --max-families 1

babappa summarize-empirical-pilot-panel \
  --panel-run real_empirical_pilot/babappa_run_hsp70_candidate \
  --outdir real_empirical_pilot/summary_hsp70_candidate

babappa validate-empirical-pilot-summary \
  --summary-dir real_empirical_pilot/summary_hsp70_candidate

echo "Done. Inspect:"
echo "  real_empirical_pilot/babappa_run_hsp70_candidate/panel_run_report.md"
echo "  real_empirical_pilot/summary_hsp70_candidate/empirical_pilot_panel_summary.md"
