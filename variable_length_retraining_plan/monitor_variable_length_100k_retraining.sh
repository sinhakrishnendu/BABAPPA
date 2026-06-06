#!/usr/bin/env bash
set -euo pipefail
WORKSPACE="${BABAPPA_RETRAIN_WORKSPACE:-branch_site_v2_100k_workspace}"
echo "Disk usage:"
df -h .
du -sh "$WORKSPACE" 2>/dev/null || true
echo "Latest retraining log:"
ls -t variable_length_retraining_plan/logs/variable_length_retraining_*.log 2>/dev/null | head -1 | xargs tail -80 2>/dev/null || true
echo "Stage markers:"
find variable_length_retraining_plan/stage_markers -type f 2>/dev/null | wc -l
