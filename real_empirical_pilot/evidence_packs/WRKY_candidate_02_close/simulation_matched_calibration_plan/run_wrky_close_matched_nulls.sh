#!/usr/bin/env bash
set -euo pipefail

echo 'USER-RUN ONLY - DO NOT EXECUTE IN CODEX'
echo 'Small pilot null calibration scaffold for WRKY_candidate_02_close.'

REPO_ROOT="${BABAPPA_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
cd "$REPO_ROOT"

# Safe dry-run preview:
# babappa run-simulation-matched-null-calibration \
#   --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close \
#   --outdir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100_dryrun \
#   --n-null 100 \
#   --seed 20260530 \
#   --device mps \
#   --dry-run

babappa run-simulation-matched-null-calibration \
  --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close \
  --outdir real_empirical_pilot/calibration_runs/WRKY_candidate_02_close_null100 \
  --n-null 100 \
  --seed 20260530 \
  --device mps
