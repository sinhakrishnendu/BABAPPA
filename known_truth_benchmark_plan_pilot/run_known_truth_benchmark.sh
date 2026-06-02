#!/usr/bin/env bash
set -euo pipefail
cd "$(pwd)"
echo 'USER-RUN ONLY — DO NOT EXECUTE IN CODEX'
mkdir -p known_truth_benchmark_pilot
babappa simulate-known-truth-benchmark \
  --design-dir known_truth_benchmark_design_v1 \
  --profile pilot \
  --outdir known_truth_benchmark_pilot/simulated_families \
  --seed 42
babappa run-known-truth-alignments \
  --sim-dir known_truth_benchmark_pilot/simulated_families \
  --outdir known_truth_benchmark_pilot/alignments \
  --methods identity,mafft,babappalign,muscle \
  --threads 8 \
  --max-workers 4
babappa score-known-truth-benchmark \
  --sim-dir known_truth_benchmark_pilot/simulated_families \
  --alignment-dir known_truth_benchmark_pilot/alignments \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir known_truth_benchmark_pilot/babappa_scores \
  --device auto \
  --score-backend direct
babappa evaluate-known-truth-benchmark \
  --truth known_truth_benchmark_pilot/simulated_families/benchmark_truth_manifest.tsv \
  --scores known_truth_benchmark_pilot/babappa_scores \
  --outdir known_truth_benchmark_pilot/evaluation
babappa evaluate-known-truth-calibration \
  --truth known_truth_benchmark_pilot/simulated_families/benchmark_truth_manifest.tsv \
  --scores known_truth_benchmark_pilot/babappa_scores \
  --outdir known_truth_benchmark_pilot/calibration_evaluation
babappa make-known-truth-benchmark-report \
  --benchmark-dir known_truth_benchmark_pilot \
  --outdir known_truth_benchmark_pilot/report
