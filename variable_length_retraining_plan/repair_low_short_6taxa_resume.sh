#!/usr/bin/env bash
set -euo pipefail

# Local resume repair script. Review before running.
# Purpose: reset only the stale low_short_6taxa alignment/downstream outputs
# that were generated when external aligners were not visible. The completed
# simulation directory is preserved.

WORKSPACE="${BABAPPA_RETRAIN_WORKSPACE:-branch_site_v2_100k_workspace}"
PLAN_DIR="${BABAPPA_RETRAIN_PLAN_DIR:-variable_length_retraining_plan}"
CHUNK="low_short_6taxa"
STAMP="$(date +%Y%m%d_%H%M%S)"
STALE_ROOT="${WORKSPACE}/stale_${CHUNK}_${STAMP}"

paths=(
  "${WORKSPACE}/align_${CHUNK}"
  "${WORKSPACE}/site_map_${CHUNK}"
  "${WORKSPACE}/method_policy_${CHUNK}"
  "${WORKSPACE}/tensors_${CHUNK}"
  "${WORKSPACE}/dataset_${CHUNK}"
  "${WORKSPACE}/labels_${CHUNK}"
  "${WORKSPACE}/branch_dataset_${CHUNK}"
)

markers=(
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_align"
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_align_methods"
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_site_map"
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_method_policy"
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_tensors"
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_index"
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_labels"
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_branch_dataset"
  "${PLAN_DIR}/stage_markers/.stage_complete_${CHUNK}_validate_branch_dataset"
)

echo "Resetting stale resume state for ${CHUNK}"
echo "Preserving ${WORKSPACE}/sim_${CHUNK}"

if [ "${BABAPPA_REPAIR_DELETE_STALE:-NO}" = "YES" ]; then
  for path in "${paths[@]}"; do
    [ -e "$path" ] && rm -rf "$path"
  done
else
  mkdir -p "$STALE_ROOT"
  for path in "${paths[@]}"; do
    if [ -e "$path" ]; then
      mv "$path" "$STALE_ROOT/"
    fi
  done
  echo "Stale outputs moved to ${STALE_ROOT}"
  echo "After confirming the rerun works, remove that stale directory manually if you want the space back."
fi

for marker in "${markers[@]}"; do
  [ -e "$marker" ] && rm -f "$marker"
done

echo "Repair complete. Rerun:"
echo "BABAPPA_RETRAIN_CLEANUP_MODE=delete BABAPPA_RETRAIN_DELETE_INTERMEDIATES=YES BABAPPA_DEVICE=mps BABAPPA_THREADS=18 bash ${PLAN_DIR}/run_variable_length_100k_retraining.sh"
