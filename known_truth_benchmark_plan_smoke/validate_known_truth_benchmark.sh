#!/usr/bin/env bash
set -euo pipefail
babappa validate-known-truth-benchmark \
  --benchmark-dir known_truth_benchmark_smoke \
  --outdir known_truth_benchmark_smoke/validation
