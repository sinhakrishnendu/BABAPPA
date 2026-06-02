#!/usr/bin/env bash
set -euo pipefail
babappa make-known-truth-benchmark-report \
  --benchmark-dir known_truth_benchmark_smoke \
  --outdir known_truth_benchmark_smoke/report
