#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY — DO NOT EXECUTE IN CODEX'
babappa make-known-truth-benchmark-report \
  --benchmark-dir known_truth_benchmark_pilot \
  --outdir known_truth_benchmark_pilot/report
