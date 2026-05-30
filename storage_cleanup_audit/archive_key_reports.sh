#!/usr/bin/env bash
set -euo pipefail
cd "$HOME/Documents/GitHub/BABAPPA"

echo "USER-RUN ONLY -- creates a compact reports/manifests archive"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="BABAPPA_KEY_REPORTS_AND_MANIFESTS_${STAMP}.tar.gz"
tar -czf "$OUT" \
  README.md docs examples \
  deployable_model_conservative_branch_site_100k_mps/model_manifest.json \
  deployable_model_conservative_branch_site_100k_mps/model_card.md \
  deployable_model_conservative_branch_site_100k_mps/feature_schema.json \
  deployable_model_conservative_branch_site_100k_mps/calibration_schema.json \
  deployable_model_conservative_branch_site_100k_mps/training_envelope.json \
  deployable_model_conservative_branch_site_100k_mps/checksums.sha256 \
  explicit_branch_truth_100k_mps_final_validation_report.md \
  explicit_branch_truth_100k_mps_final_validation_report.json \
  explicit_branch_truth_100k_mps_final_validation_report.tsv \
  explicit_branch_truth_100k_mps_cross_tier_summary \
  branch_truth_status_audit_explicit_branch_truth_100k_mps \
  real_empirical_pilot/evidence_packs/WRKY_candidate_02_close \
  real_empirical_pilot/summary \
  real_empirical_pilot/reference_results \
  storage_cleanup_audit
shasum -a 256 "$OUT" > "${OUT}.sha256"
echo "Archive: $OUT"
