#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY — DO NOT EXECUTE IN CODEX'
babappa plan-known-truth-reference-comparison \
  --benchmark-dir known_truth_benchmark_extended \
  --outdir known_truth_benchmark_extended/reference_comparison_plan \
  --tools codeml,absrel,meme \
  --max-families 100
