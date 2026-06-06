#!/usr/bin/env bash
set -euo pipefail
echo "Sensitivity-analysis plan script: this writes a checklist and does not retrain models."
GRID="${1:-sensitivity_analysis_grid.tsv}"
OUTDIR="${2:-sensitivity_analysis_results}"
mkdir -p "$OUTDIR"
cp "$GRID" "$OUTDIR/sensitivity_analysis_grid.tsv"
cat > "$OUTDIR/sensitivity_analysis_readme.md" <<'EOF'
# Sensitivity Analysis Checklist

Run the axes in `sensitivity_analysis_grid.tsv` only when the required model-training or calibration resources are available. Record AUROC, AUPRC, FDR, MCC, OOD false-call rate, and threshold stability for each axis.
EOF
