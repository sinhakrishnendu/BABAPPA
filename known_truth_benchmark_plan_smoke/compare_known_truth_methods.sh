#!/usr/bin/env bash
set -euo pipefail
babappa plan-known-truth-reference-comparison \
  --benchmark-dir known_truth_benchmark_smoke \
  --outdir known_truth_benchmark_smoke/reference_comparison_plan \
  --tools codeml,absrel,meme \
  --max-families 100
