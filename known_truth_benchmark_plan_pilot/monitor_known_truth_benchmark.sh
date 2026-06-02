#!/usr/bin/env bash
set -euo pipefail
echo 'USER-RUN ONLY — DO NOT EXECUTE IN CODEX'
du -sh known_truth_benchmark_pilot 2>/dev/null || true
find known_truth_benchmark_pilot -maxdepth 3 -type f | wc -l
find known_truth_benchmark_pilot -maxdepth 3 -name '*summary*.json' -o -name '*manifest*.json' 2>/dev/null | sort
