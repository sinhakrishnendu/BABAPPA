#!/usr/bin/env bash
set -euo pipefail
echo "Retained validation profile: review disk budget before running."
OUTDIR="${1:-benchmark_runs/retained_validation_profile}"
ROOT_DIR="${BABAPPA_REPO_ROOT:-$(pwd)}"
mkdir -p "$OUTDIR"
CONFIG="$OUTDIR/config_retained_validation.yaml"
cat > "$CONFIG" <<EOF
profile: retained_validation
n_families: 10000
seed: 20260605
n_taxa: 10
n_codons: 240
outdir: $OUTDIR
model_package: deployable_model_conservative_branch_site_100k_mps
device: auto
babappa_null_replicates: 100
jobs:
  babappa: 8
  absrel: 8
  prepare: 8
EOF
cat > "$OUTDIR/retained_validation_run_plan.md" <<'EOF'
# Retained Validation Profile

This profile should retain all compact inputs, truth files, features, scores, summaries, manifests, and checksums. It is intended to repair the reproducibility limitation from the pruned 100K intermediates.
EOF
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/01_simulate_known_truth_dataset.py" --config "$CONFIG"
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/02_run_babappa_on_dataset.py" --config "$CONFIG" --continue-on-failure --jobs "8"
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/03_prepare_absrel_inputs.py" --config "$CONFIG" --jobs "8"
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/05_compare_against_truth.py" --config "$CONFIG"
python "$ROOT_DIR/scripts/benchmarks/known_truth_absrel/06_make_benchmark_report.py" --config "$CONFIG"
