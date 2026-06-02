#!/usr/bin/env bash
set -euo pipefail
cd "$(pwd)"
mkdir -p known_truth_benchmark_smoke
babappa simulate-known-truth-benchmark \
  --design-dir known_truth_benchmark_design_v1 \
  --profile smoke \
  --outdir known_truth_benchmark_smoke/simulated_families \
  --seed 42
babappa run-known-truth-alignments \
  --sim-dir known_truth_benchmark_smoke/simulated_families \
  --outdir known_truth_benchmark_smoke/alignments \
  --methods identity,mafft,babappalign,muscle \
  --threads 2 \
  --max-workers 1
babappa score-known-truth-benchmark \
  --sim-dir known_truth_benchmark_smoke/simulated_families \
  --alignment-dir known_truth_benchmark_smoke/alignments \
  --deployable-model-package deployable_model_conservative_branch_site_100k_mps \
  --outdir known_truth_benchmark_smoke/babappa_scores \
  --device auto \
  --score-backend smoke_surrogate
babappa evaluate-known-truth-benchmark \
  --truth known_truth_benchmark_smoke/simulated_families/benchmark_truth_manifest.tsv \
  --scores known_truth_benchmark_smoke/babappa_scores \
  --outdir known_truth_benchmark_smoke/evaluation
babappa evaluate-known-truth-calibration \
  --truth known_truth_benchmark_smoke/simulated_families/benchmark_truth_manifest.tsv \
  --scores known_truth_benchmark_smoke/babappa_scores \
  --outdir known_truth_benchmark_smoke/calibration_evaluation
babappa make-known-truth-benchmark-report \
  --benchmark-dir known_truth_benchmark_smoke \
  --outdir known_truth_benchmark_smoke/report
