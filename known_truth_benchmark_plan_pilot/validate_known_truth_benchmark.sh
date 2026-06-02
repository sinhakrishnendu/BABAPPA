#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY — DO NOT EXECUTE IN CODEX'
babappa validate-known-truth-benchmark \
  --benchmark-dir known_truth_benchmark_pilot \
  --outdir known_truth_benchmark_pilot/validation
