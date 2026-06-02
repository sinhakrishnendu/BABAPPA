#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/GitHub/BABAPPA"

echo "MANUAL EXECUTION SCRIPT -- lightweight validation after quarantine"
python -m pip install -e ".[dev]"
python -m pytest -q
babappa validate-deployable-model-package --package-dir deployable_model_conservative_branch_site_100k_mps
if command -v babappa >/dev/null 2>&1 && [ -d real_empirical_pilot/evidence_packs/WRKY_candidate_02_close ]; then
  babappa validate-empirical-evidence-pack --evidence-pack real_empirical_pilot/evidence_packs/WRKY_candidate_02_close || echo "WARNING: evidence-pack validation reported a problem"
else
  echo "WARNING: evidence-pack validator or WRKY evidence pack unavailable"
fi
du -sh .
git status --short
